"""HTTP API and static web UI.

The JSON contract lives in the prototype spec ("HTTP API") and is mirrored by web/*.js.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import math
import os
import re
import time
import unicodedata
import uuid
from collections.abc import Callable, Coroutine, Iterator
from contextlib import aclosing, asynccontextmanager, contextmanager
from dataclasses import replace
from pathlib import Path, PurePath
from typing import Annotated, Any, Literal
from urllib.parse import quote

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, StringConstraints
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import contributors, csp
from .anchor import read_anchor
from .config import Settings
from .consent import MANAGE_TOKEN_PATTERN, PUBLIC_ID_HEX, bound_salt, check_manage_token, manage_token, public_id
from .contributors import MAX_NICKNAME_CHARS
from .coverage import compute_coverage
from .gzipfiles import GzipCache, GzipFiles
from .ingest import UploadOutcome, ingest_upload, pin_malloc_threshold
from .jobs import JobRunner, recorded_snap, shown_localizations, stylized_snaps
from .photoposes import (POSES_FORMAT, POSES_VERSION, PUBLIC_PHOTO_VERSION, processed_aspects, public_photo,
                         public_poses, public_rounding, public_uncertainty, snap_accepted)
from .reconstruct import Reconstructor, get_reconstructor
from .records import (
    ACTIVE_JOB_STATUSES,
    JOB_DONE,
    JOB_FACE,
    JOB_FAILED,
    JOB_FULL,
    JOB_LOCALIZE,
    JOB_STYLIZE,
    PHOTO_ACCEPTED,
    TERMS_CURRENT,
    Contributor,
    Job,
    Landmark,
    ModelRecord,
    PhotoRecord,
    location_shared,
)
from .store import AlreadyExists, NotFound, Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
LANDMARK_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"
MAX_FILES_PER_UPLOAD = 50
DETAIL_MAX_JOBS = 20
# Served free text (a job's error, a stylized model's registration reason) may quote a server path, whose file name is
# a store photo id (data/photos/<landmark>/<photo_id>.proc.jpg), or name a photo by its id: public_text replaces both
# (D30).
_PATH_IN_TEXT = re.compile(r"(?<![\w.~-])/(?:[^\s'\"`,;:()<>\[\]{}/]+/)+[^\s'\"`,;:()<>\[\]{}]*")
# A hex run, also inside a file name (COLMAP's images are 00001_<photo_id>.jpg); a model id after "model " is public.
_HEX_ID_IN_TEXT = re.compile(r"(?<!model )(?<![0-9a-fA-F])[0-9a-f]{32,}(?![0-9a-fA-F])")
TERMS_URL = "/terms.html"            # the user agreement (D32), served from the web dir
REGISTER_ROUTE = "/api/contributors"            # D33: register once, agreeing to the user agreement
ACCEPT_ROUTE = "/api/contributors/accept"       # D33: a registered contributor accepts a new agreement version
UPLOAD_ROUTE = "/api/photos"                    # D33: its form holds the contributor token; no 422 of it echoes input
# (D34: the one route with a large body, BodySizeLimit, and with a form read within its own limits, UploadRoute)
# D34: a contributor can always withdraw their own photos. The withdrawal also takes the registration (its body may
# hold the contributor token, so no 422 of it echoes input either), and a registration lists its own shared photos.
WITHDRAW_ROUTE = "/api/photos/{photo_id}/share_location"
OWN_PHOTOS_ROUTE = "/api/contributors/photos"
NO_ECHO_ROUTES = (UPLOAD_ROUTE, WITHDRAW_ROUTE)   # routes FastAPI parses whose body may hold a token (validation_error)
# The one 403 of a withdrawal, whatever was wrong: the manage token, the contributor token, or whose photo it is
WITHDRAWAL_REFUSED = "wrong token for this photo"
TOKEN_FIELD = "contributor_token"    # the contributor token's field, only ever in a POST body (never a URL or header)
JSON_MEDIA_TYPE = "application/json"
BODY_CUT_OFF = "the request body was cut off"     # the 400 of a contributor body whose client left before its end
# The body of a registration or an acceptance (D33): a nickname of at most 64 characters, a version and a token fit in
# well under 1 KiB; a bigger body is refused unread (contributor_fields)
CONTRIBUTOR_BODY_MAX_BYTES = 16 * 1024
_CONTENT_LENGTH = re.compile(r"[0-9]{1,9}")     # ASCII digits only (str.isdigit takes "²", which int() refuses)
_PUBLIC_ID = re.compile(rf"[0-9a-f]{{{PUBLIC_ID_HEX}}}")      # a public row id (consent.public_id)
NO_SUCH_PHOTO = "no such photo"      # the one 404 of GET .../poses/{public_id}/photo, whatever the reason
# The served photo (D32): browsers may keep it, but ask again on every view (If-None-Match), so a withdrawal counts at
# once; no Last-Modified (the processed copy's mtime is the upload time, D30)
PHOTO_CACHE_CONTROL = "no-cache"
DEMO_LANDMARK_ID = "demo-building"
DEMO_RADIUS_M = 150.0
R_SERVER_ERROR = "server_error"       # upload result: storing this file failed on the server (disk full, ...)
BODY_SLACK_BYTES = 1024 * 1024        # multipart headers and form fields on top of the files
# The body of every request but the upload (D34): the JSON bodies hold a few short fields. Only POST UPLOAD_ROUTE may
# be as big as its files (BodySizeLimit); the contributor routes read at most CONTRIBUTOR_BODY_MAX_BYTES themselves.
SMALL_BODY_MAX_BYTES = 64 * 1024
# The upload's form (D34, UploadRequest): what it may hold besides the bytes of its files. Its own text fields are a
# token and a landmark id, well under 100 bytes each (a D32 client sent two more), and the parser keeps text fields in
# memory: Starlette's own limits (1000 fields of 1 MiB, 1000 files, a file part under 1 MiB kept in memory) let one
# request without a token hold 1 GB there. With these, a form holds at most 32 KiB of text and 3.2 MB of small files
# in memory; a bigger file part goes to a temporary file as before.
UPLOAD_MAX_FIELDS = 8                 # text fields
UPLOAD_FIELD_MAX_BYTES = 4 * 1024     # of one text field
UPLOAD_SPOOL_BYTES = 64 * 1024        # of one file part kept in memory
MULTIPART_MEDIA_TYPE = "multipart/form-data"
# What a 422 carries back of one error's input, as the JSON the response holds: more than this and the input is null
# (D34: an error names its field and type; the client has what it sent)
MAX_ECHO_BYTES = 1024
MAX_ECHO_CHARS = 80                   # of a path or form value quoted in a 404 (a landmark id is at most 64)
# On every response (D34, BaselineHeaders). nosniff: a response is only what its Content-Type says. X-Frame-Options:
# no other site shows these pages in a frame (clickjacking); the pages do not frame each other. Referrer-Policy: other
# sites get the origin at most, never a page's path or query (the landmark, the model). Not no-referrer or
# same-origin: the home page's map tiles come from tile.openstreetmap.org, whose usage policy requires pages to send
# a Referer and blocks traffic that strips it. Content-Security-Policy: a response that sets none gets the strict
# one (csp.STRICT_POLICY: it may load nothing); only the site's pages set their own (WebFiles, csp.page_policy).
BASELINE_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (csp.HEADER.lower().encode(), csp.STRICT_POLICY.encode()),
)
# The web UI's own files (WebFiles): a browser keeps them but asks again on every use (the ETag answers 304), so a
# returning visitor never runs a new module with an old one after a deployment (the modules have no versioned URLs)
WEB_CACHE_CONTROL = "no-cache"
WEB_PAGE_SUFFIXES = (".html", ".htm")   # the site's pages: each is served with its own Content-Security-Policy
WEB_NOT_SERVED = ("tests",)           # top-level folders of the web dir that are not part of the site
# The site's files sent gzip-compressed to a client that accepts it (D34, gzipfiles.GzipFiles): its text, the
# vendored libraries among it. Nothing else: the libraries' PNG images are compressed already.
WEB_GZIP_SUFFIXES = (".html", ".htm", ".js", ".mjs", ".css", ".json", ".svg")
# The API sends codes and technical English only; the web client words them in the page language (D32). A stylized
# model's label (web/models.js stylizedLabel: always clay, D29) comes from its registration.accepted, which decides
# whether photo evidence (D15, D24, D25) may change the clay model.

log = logging.getLogger(__name__)


class LandmarkIn(BaseModel):
    id: Annotated[str, Field(pattern=LANDMARK_ID_PATTERN)] | None = None
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    lat: Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
    lon: Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]
    radius_m: Annotated[float, Field(gt=0, le=5000, allow_inf_nan=False)] | None = None


class RebuildIn(BaseModel):
    kind: Literal["full", "face"]
    sector: int | None = None


class RegisterIn(BaseModel):
    """Register a contributor (D33): the nickname their accepted photos carry, and the user agreement version they
    agree to. The documented body only (OpenAPI): the route reads its body by hand (contributor_fields) and checks
    both fields by hand (terms first), so a refusal has one code and one order."""
    nickname: str
    terms: str


class AcceptIn(BaseModel):
    """A registered contributor accepts the current user agreement (D33). The documented body only (OpenAPI): the
    route reads its body by hand (contributor_fields), so no refusal echoes any part of the token."""
    contributor_token: str
    terms: str


def _json_body_doc(model: type[BaseModel]) -> dict[str, Any]:
    """openapi_extra of a route that reads its JSON body by hand: the body it expects, for the API docs."""
    return {"requestBody": {"required": True, "content": {JSON_MEDIA_TYPE: {"schema": model.model_json_schema()}}}}


async def contributor_fields(request: Request) -> dict[str, Any]:
    """The fields of a registration's or an acceptance's body (D33), read by hand. FastAPI's own 422 for a body of
    the wrong shape echoes what was sent (string_type with the list or object, model_attributes_type with the raw
    bytes of a body that is not JSON), and the body of an acceptance holds the contributor token. Anything but a JSON
    object sent as application/json in at most CONTRIBUTOR_BODY_MAX_BYTES (not JSON, a form, a JSON array, string,
    number or null, any other or no content type, a bigger body) counts as an empty object, so the route refuses it
    with its own codes (terms_required, register_required), which echo nothing of it. A client that leaves before
    the end of its body gets a quiet 400 BODY_CUT_OFF (as FastAPI's own body parsing did), not a logged traceback."""
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != JSON_MEDIA_TYPE:
        return {}
    length = request.headers.get("content-length")
    if length is not None and not (_CONTENT_LENGTH.fullmatch(length) and int(length) <= CONTRIBUTOR_BODY_MAX_BYTES):
        return {}
    body = bytearray()
    try:
        async for chunk in request.stream():
            body += chunk
            if len(body) > CONTRIBUTOR_BODY_MAX_BYTES:
                return {}
    except ClientDisconnect:
        raise HTTPException(400, BODY_CUT_OFF) from None
    try:
        fields = json.loads(body)
    except (ValueError, RecursionError):     # not JSON, not UTF-8 (UnicodeDecodeError is a ValueError), nested deep
        return {}
    return fields if isinstance(fields, dict) else {}


class ShareLocationIn(BaseModel):
    """Withdraw a photo (D30, D32): hides its public pose and its photo. The user agreement is agreed to only at
    registration (D33), so share is always false. By one of two tokens (D34), never both: token, the photo's manage
    token, or contributor_token, the token of the registration the photo was uploaded under."""
    share: Literal[False]
    token: Annotated[str, Field(pattern=MANAGE_TOKEN_PATTERN)] | None = None
    contributor_token: str | None = None


class OwnPhotosIn(BaseModel):
    """List a registration's own shared photos (D34). The documented body only (OpenAPI): the route reads its body
    by hand (contributor_fields), so no refusal echoes any part of the token."""
    contributor_token: str
    landmark_id: str | None = None


def slugify(name: str) -> str:
    """ASCII landmark id from a name; "lm-" + 8 hex chars when nothing ASCII is left (e.g. Chinese names)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:64].rstrip("-")
    return slug or f"lm-{uuid.uuid4().hex[:8]}"


class DataDirInUse(RuntimeError):
    pass


@contextmanager
def data_dir_lock(settings: Settings) -> Iterator[None]:
    """Exclusive lock on the data dir for the life of the server.

    The job runner re-queues every job left "running" when it starts, so a
    second server on the same data dir (an accidental second launch, a rolling
    deploy) would run the first one's jobs again in the same work dirs. It must
    fail before touching anything; uvicorn runs the lifespan before binding.
    """
    try:
        import fcntl
    except ImportError:      # not on POSIX: no protection
        yield
        return
    fd = os.open(settings.data_dir / "crowd3d.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DataDirInUse(f"another Crowd3D server is using the data dir {settings.data_dir}") from None
        yield
    finally:
        os.close(fd)


class BodySizeLimit:
    """413 for a request body above its route's limit, before anything parses it (D34): upload_max_bytes for the
    upload (POST UPLOAD_ROUTE: before multipart parsing spools it to temp files), max_bytes for every other request
    (FastAPI reads a JSON body whole into memory before it validates it). Decided from Content-Length when the client
    sends one, and by counting the body as it arrives otherwise.

    A body counted as it arrives can pass the limit after the application began its answer: a file response waits on
    receive() for the client leaving while it sends the file (Starlette's FileResponse, both file mounts), so a GET
    for a file that carries a chunked body may have its 200 on the way by then. No 413 can follow a response start
    (the server raises on a second one), and an answer cut short is an error in the server's log as well. So from the
    response start on, what arrives over the limit is read and dropped, and the answer is finished: the application
    never gets more than the limit, and it is still told when the client leaves. Such a request gets the 413 or the
    file, by whether its body or the answer came first. (The compressed form of a file reads the request's body
    before it answers, gzipfiles.GzipFileResponse: over the limit, that request always gets the 413.)"""

    def __init__(self, app: ASGIApp, max_bytes: int, upload_max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.upload_max_bytes = upload_max_bytes

    def limit(self, scope: Scope) -> int:
        upload = scope.get("method") == "POST" and _route_path(scope) == UPLOAD_ROUTE
        return self.upload_max_bytes if upload else self.max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit(scope)
        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            await self._reject(send, limit)
            return
        # exceeded: the body passed the limit before the answer began, so the answer is the 413 (rejected: sent)
        received, exceeded, started, rejected = 0, False, False, False

        async def counted_receive() -> Message:
            nonlocal received, exceeded
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    return message
                received += len(message.get("body", b""))
                if received <= limit:
                    return message
                if not started:
                    exceeded = True
                    raise _BodyTooLarge
                # over the limit with the answer begun: dropped, and on to the next message

        async def guarded_send(message: Message) -> None:
            nonlocal started, rejected
            if not exceeded:
                # noted before the await: the start is with the server from here on, also while this send waits
                started = started or message["type"] == "http.response.start"
                await send(message)
            elif not rejected:
                # FastAPI turns the aborted body parsing into its own 400; answer 413 instead.
                rejected = True
                await self._reject(send, limit)

        try:
            await self.app(scope, counted_receive, guarded_send)
        except _BodyTooLarge:       # raised only before the answer began, and nothing is forwarded after it
            if not rejected:
                rejected = True
                await self._reject(send, limit)

    async def _reject(self, send: Send, limit: int) -> None:
        response = JSONResponse({"detail": f"request body over {limit} bytes"}, status_code=413)
        await send({"type": "http.response.start", "status": 413, "headers": response.raw_headers})
        await send({"type": "http.response.body", "body": response.body})


class _BodyTooLarge(Exception):
    pass


def _route_path(scope: Scope) -> str:
    """The request path as the routes see it: without the root path a proxy mounts the app under."""
    path, root = scope["path"], scope.get("root_path", "")
    if root and path.startswith(root) and path[len(root):].startswith("/"):
        return path[len(root):]
    return path


def _declared_length(scope: Scope) -> int | None:
    """The request's Content-Length (the largest, if it sends several); None without one or when it is not a
    number, and the body is then counted as it arrives."""
    declared = None
    for name, value in scope.get("headers") or ():
        if name == b"content-length":
            digits = value.strip()
            if not digits.isdigit():        # bytes: ASCII digits only
                return None
            digits = digits.lstrip(b"0")
            # more digits than any limit has: over it, without converting a number of any length
            length = int(digits or b"0") if len(digits) <= 18 else 10 ** 18
            declared = length if declared is None else max(declared, length)
    return declared


class _UploadParser(MultiPartParser):
    spool_max_size = UPLOAD_SPOOL_BYTES     # (Starlette reads this class attribute: there is no parameter for it)


class _FormRefused(HTTPException):
    """The parser refused the upload's form (UploadRequest.form): it went over one of its limits, or is malformed
    (no boundary, a part without a name). Holds the parser's message and the text fields it had read by then. An
    HTTPException, which FastAPI's body reading lets through as it is; UploadRoute answers it (_upload_refusal)."""

    def __init__(self, message: str, fields: list[tuple[str, str]]) -> None:
        super().__init__(400, message)
        self.fields = fields


class UploadRequest(Request):
    """The request of POST UPLOAD_ROUTE as its route reads it (UploadRoute; D34): the form is parsed within the
    upload's own limits, whatever limits the caller passes (FastAPI passes none, so Starlette's applied): at most
    MAX_FILES_PER_UPLOAD file parts, UPLOAD_MAX_FIELDS text fields of UPLOAD_FIELD_MAX_BYTES each, and a file part
    stays in memory up to UPLOAD_SPOOL_BYTES. The parser stops at the first part over a limit (_FormRefused), without
    reading the rest of the body. A body that is not multipart/form-data holds no file, so it is an empty form and
    none of it is read (Starlette parses a form-encoded body into memory too)."""

    def form(self, **_limits: Any) -> Coroutine[Any, Any, FormData]:    # awaited, as Starlette's own is
        return self._upload_form()

    async def _upload_form(self) -> FormData:
        if self._form is None:
            if self.headers.get("content-type", "").split(";", 1)[0].strip().lower() != MULTIPART_MEDIA_TYPE:
                self._form = FormData()
                return self._form
            async with aclosing(self.stream()) as stream:
                parser = _UploadParser(self.headers, stream, max_files=MAX_FILES_PER_UPLOAD,
                                       max_fields=UPLOAD_MAX_FIELDS, max_part_size=UPLOAD_FIELD_MAX_BYTES)
                try:
                    self._form = await parser.parse()
                except MultiPartException as exc:
                    fields = [(name, value) for name, value in parser.items if isinstance(value, str)]
                    raise _FormRefused(exc.message, fields) from None
        return self._form


class UploadRoute(APIRoute):
    """The route class of POST UPLOAD_ROUTE (D34): FastAPI reads the form from an UploadRequest, and a form over
    its limits gets the upload's own answer (_upload_refusal) instead of Starlette's 400."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def limited(request: Request) -> Response:
            try:
                return await handler(UploadRequest(request.scope, request.receive))
            except _FormRefused as refused:
                raise await run_in_threadpool(_upload_refusal, request.app.state.store, refused) from None

        return limited


class BaselineHeaders:
    """Adds BASELINE_HEADERS to every response that does not set them itself (D34). Wraps the whole application
    (_App.build_middleware_stack), so the body limit's 413, a static file, a 304 and an unhandled error's 500 carry
    them like any route's response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or ())
                sent = {name.lower() for name, _ in headers}
                headers += [(name, value) for name, value in BASELINE_HEADERS if name not in sent]
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class _App(FastAPI):
    def build_middleware_stack(self) -> ASGIApp:
        return BaselineHeaders(super().build_middleware_stack())


class ModelFiles(GzipFiles):
    """Serves only the model point clouds (.ply) and the stylized models (.glb); the sidecars next to them (.json,
    .style.json) hold per-photo camera poses. Both kinds are public and compress well (a point cloud is ASCII, a
    GLB mostly JSON), so a client that accepts gzip gets them compressed (D34, GzipFiles); a request for a range
    gets the plain bytes."""

    SUFFIXES = (".ply", ".glb")
    gzip_suffixes = SUFFIXES

    async def get_response(self, path: str, scope: Scope):
        if not path.endswith(self.SUFFIXES):
            raise HTTPException(404)
        return await super().get_response(path, scope)


class WebFiles(GzipFiles):
    """Serves the web UI (the web dir) with three rules (D34). Every file (the pages, scripts, styles and JSON, and
    the libraries in web/vendor with their images and licences) carries Cache-Control WEB_CACHE_CONTROL, on a 200 and
    on a 304 alike: StaticFiles sends an ETag and Last-Modified but no Cache-Control, so browsers kept each file for
    a time of their own guessing, and a returning visitor could run a new module with an old one. The folders in
    WEB_NOT_SERVED (the node tests) are not served: 404. And a page (WEB_PAGE_SUFFIXES) carries its own
    Content-Security-Policy (csp.page_policy: the site's own files and the page's inline blocks by hash), on a 200
    and on a 304 alike: a browser replaces the headers it stored with a 304's, so a 304 with the strict policy every
    other response gets (BaselineHeaders) would leave the stored page unable to load anything. All three go by the
    file a request resolves to, not by how its path is spelt ("/../web/tests/..." resolves into the web dir too).
    The text files (WEB_GZIP_SUFFIXES) go compressed to a client that accepts gzip (GzipFiles), with the same
    headers on either form."""

    gzip_suffixes = WEB_GZIP_SUFFIXES

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.page_policies = csp.PagePolicies()

    def lookup_path(self, path: str):
        full_path, stat_result = super().lookup_path(path)
        if stat_result is not None:
            inside = PurePath(os.path.relpath(full_path, os.path.realpath(self.directory))).parts
            if inside and inside[0].casefold() in WEB_NOT_SERVED:      # (no parts: the web dir itself)
                return "", None
        return full_path, stat_result

    def file_response(self, full_path, stat_result, scope: Scope, status_code: int = 200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = WEB_CACHE_CONTROL
        if str(full_path).lower().endswith(WEB_PAGE_SUFFIXES):
            response.headers[csp.HEADER] = self.page_policies.of(full_path, stat_result)
        return response


def create_app(settings: Settings | None = None, reconstructor: Reconstructor | None = None) -> FastAPI:
    settings = settings or Settings()
    data_dir = settings.data_dir.resolve()
    if data_dir == WEB_DIR or WEB_DIR in data_dir.parents:
        # the web dir is served as is: the database, the photos and the public-id salt would be public
        raise ValueError(f"the data dir {settings.data_dir} must not lie inside the served web dir {WEB_DIR}")
    settings.ensure_dirs()
    store = Store(settings)
    try:
        # the secret behind public photo ids and manage tokens (D30): <data_dir>/public_id.salt, never served, and
        # the one the store's consents were given under (a different one refuses to start while a photo is shared)
        salt = bound_salt(settings.data_dir, store)
    except BaseException:
        store.close()
        raise
    runner = JobRunner(store, settings, reconstructor or get_reconstructor(settings.reconstructor, settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            with data_dir_lock(settings):
                # for the process this server runs in (D34): without it the decode limit does not bound its memory
                pin_malloc_threshold(settings.malloc_mmap_threshold_bytes)
                runner.start()
                try:
                    yield
                finally:
                    runner.stop()
        finally:
            store.close()

    # No /docs and /redoc (D34): FastAPI's pages load Swagger UI and ReDoc from a CDN and run an inline script, on
    # this origin, next to the tokens the site's pages keep in localStorage. /openapi.json stays.
    app = _App(title="Crowd3D", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.store = store
    app.state.runner = runner

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI's 422, which echoes each error's input, with three changes. No 422 of the upload (UPLOAD_ROUTE)
        echoes any input (D33): its form holds the contributor token, and FastAPI's own 422 for a part of the wrong
        kind echoes the part (a file part sent as the token: its file name, part headers and temporary file; a text
        part sent as the files: the text). Nor does any 422 of the withdrawal (WITHDRAW_ROUTE, D34), whose body may
        hold the contributor token: FastAPI's own 422 for a missing field echoes the whole body. No input that cannot
        be rendered is echoed (_rendered_error), so a 422 never turns into a 500. And no input over MAX_ECHO_BYTES is
        echoed (D34): FastAPI echoes the whole body once per missing field, so a 422 was several times the size of
        what was sent."""
        errors = exc.errors()
        if getattr(request.scope.get("route"), "path", None) in NO_ECHO_ROUTES:
            errors = [{**error, "input": None} for error in errors]
        return JSONResponse(status_code=422, content={"detail": [_rendered_error(error) for error in errors]})

    def get_landmark(landmark_id: str) -> Landmark:
        try:
            return store.get_landmark(landmark_id)
        except NotFound:
            raise HTTPException(404, f"unknown landmark {_quoted(landmark_id)}") from None

    @app.get("/api/health")
    def health():
        return {"ok": True, "reconstructor": runner.reconstructor.name}

    @app.get("/api/landmarks")
    def list_landmarks():
        return {"landmarks": [landmark_summary(store, settings, lm) for lm in store.list_landmarks()]}

    @app.post("/api/landmarks", status_code=201)
    def create_landmark(body: LandmarkIn):
        lm = Landmark(
            id=body.id or slugify(body.name),
            name=body.name,
            lat=body.lat,
            lon=body.lon,
            radius_m=settings.default_radius_m if body.radius_m is None else body.radius_m,
            created_at=time.time(),
        )
        try:
            lm = store.create_landmark(lm)
        except AlreadyExists:
            raise HTTPException(409, f"landmark {lm.id!r} already exists") from None
        return landmark_summary(store, settings, lm)

    @app.get("/api/landmarks/{landmark_id}")
    def landmark_detail_route(landmark_id: str):
        return landmark_detail(store, settings, get_landmark(landmark_id))

    @app.get("/api/landmarks/{landmark_id}/poses")
    def landmark_poses_route(landmark_id: str):
        """Where the landmark's photos were taken and which way they looked, for the shared photos (D30, D32: uploaded
        under the user agreement and not withdrawn): one audience, no parameters (landmark_poses)."""
        return landmark_poses(store, settings, get_landmark(landmark_id), salt)

    @app.get("/api/landmarks/{landmark_id}/poses/{public_id}/photo")
    def landmark_photo_route(landmark_id: str, public_id: str, request: Request):
        """The photo of one public poses row (D32), keyed by the row's public id: public_photo of its processed copy
        (no metadata, longer side at most 1600 px, JPEG) while it is a row of the landmark's public poses now
        (landmark_poses: the snapshot, shared, a determined pose) and, read again, still shared; the same 404 for
        everything else (a withdrawn, rejected, later or unplaced photo, a store id, another landmark's row, any other
        string). Cache-Control no-cache and a weak ETag of the public id: a browser keeps the bytes but asks every time,
        so a withdrawal hides the photo at once. No Last-Modified: the file's time is the upload time (D30)."""
        photo = shared_photo(store, settings, get_landmark(landmark_id), salt, public_id)
        path = settings.data_dir / photo.processed_path if photo is not None and photo.processed_path else None
        if path is None or not path.is_file():
            if path is not None:
                log.warning("shared photo %s has no processed copy at %s", photo.id, path)
            raise HTTPException(404, NO_SUCH_PHOTO)
        etag = f'W/"{public_id}.{PUBLIC_PHOTO_VERSION}"'
        headers = {"Cache-Control": PHOTO_CACHE_CONTROL, "ETag": etag}
        if etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
        try:
            data = public_photo(path)
        except (OSError, ValueError):
            log.warning("cannot read the processed copy of shared photo %s at %s", photo.id, path, exc_info=True)
            raise HTTPException(404, NO_SUCH_PHOTO) from None
        return Response(data, media_type="image/jpeg", headers={**headers, "X-Content-Type-Options": "nosniff"})

    @app.post(REGISTER_ROUTE, status_code=201, openapi_extra=_json_body_doc(RegisterIn))
    def register_contributor(body: Annotated[dict[str, Any], Depends(contributor_fields)], response: Response):
        """Register a contributor (D33): the user agreement is accepted here, once, not with every upload. Terms
        first (terms_required / terms_outdated), then the nickname (1-64 characters, no control characters or lone
        surrogates, any script; not unique). Returns the secret contributor token, this once (Cache-Control:
        no-store); the store keeps only its hash. Never the contributor's id. No GET: nothing of a contributor is
        served. The body is read by hand (contributor_fields): a refusal echoes nothing but a nickname that is text
        (that encodes to UTF-8: _is_text)."""
        terms = body.get("terms")
        _check_terms(terms, "registering")
        sent = body.get("nickname")
        nickname = contributors.clean_nickname(sent)
        if nickname is None:
            raise _invalid("nickname", f"must be 1-{MAX_NICKNAME_CHARS} characters without control characters",
                           sent if _is_text(sent) else None)
        contributor, token = contributors.register(store, nickname, terms)
        response.headers["Cache-Control"] = "no-store"
        return {"nickname": contributor.nickname, "terms": contributor.terms, TOKEN_FIELD: token}

    @app.post(ACCEPT_ROUTE, openapi_extra=_json_body_doc(AcceptIn))
    def accept_terms(body: Annotated[dict[str, Any], Depends(contributor_fields)]):
        """A registered contributor accepts the current user agreement (D33: after a new version, once; no new
        nickname): register_required for a missing, malformed or unknown token, then terms_required /
        terms_outdated. The body is read by hand (contributor_fields), and no refusal echoes any of it: whatever the
        body's shape, no part of the token comes back."""
        contributor = contributors.authenticate(store, body.get(TOKEN_FIELD))
        if contributor is None:
            raise _register_required("accepting")
        terms = body.get("terms")
        _check_terms(terms, "accepting")
        contributor = contributors.accept(store, contributor, terms)
        return {"nickname": contributor.nickname, "terms": contributor.terms}

    @app.post(OWN_PHOTOS_ROUTE, openapi_extra=_json_body_doc(OwnPhotosIn))
    def own_photos(body: Annotated[dict[str, Any], Depends(contributor_fields)], response: Response):
        """A registration's own photos that are shared now (D34): {contributor_token, landmark_id?} -> {"photos":
        [{photo_id, landmark_id, public_id, manage_token}]}, in upload order, at every landmark or at landmark_id
        only (an id no landmark has lists nothing). What the uploader's page needs to show and withdraw them, and no
        more: a manage token used to exist only in its upload's response, so a response that never arrived (a dropped
        connection, an error after the photos were stored), another tab or cleared site data left a public photo
        nobody could withdraw. Only ever the photos uploaded under this registration (photos.contributor_id), and
        only the shared ones (records.location_shared): never a rejected or withdrawn photo, never anything of
        another registration. register_required for a missing, malformed or unknown token. The accepted agreement
        version is not checked: finding and withdrawing one's photos must not take accepting a new agreement. The
        body is read by hand (contributor_fields) and no refusal echoes any of it; the answer holds manage tokens:
        Cache-Control: no-store. No GET: the token travels in the POST body only."""
        contributor = contributors.authenticate(store, body.get(TOKEN_FIELD))
        if contributor is None:
            raise _register_required("listing photos")
        landmark_id = body.get("landmark_id")
        if landmark_id is not None and not _is_text(landmark_id):
            raise _invalid("landmark_id", "must be a landmark id, or left out for every landmark", None)
        photos = [p for p in store.photos_of_contributor(contributor.id, landmark_id) if location_shared(p)]
        response.headers["Cache-Control"] = "no-store"
        return {"photos": [own_photo(p, salt) for p in photos]}

    uploads = APIRouter(route_class=UploadRoute)

    @uploads.post(UPLOAD_ROUTE)
    def upload_photos(
        files: Annotated[list[UploadFile], File()],
        # D33: the registered contributor's token (POST /api/contributors). No constraint: one that is not a token is
        # register_required, input null; and no 422 of this route echoes any input, whatever the form (validation_error)
        contributor_token: Annotated[str | None, Form()] = None,
        landmark_id: Annotated[str | None, Form()] = None,
    ):
        # D33: no upload without a registered contributor who accepted the current user agreement, checked first:
        # nothing is stored, no job planned. The nickname and the consent come from the contributor's record. The
        # codes and the English text are technical; clients translate by type.
        contributor = _uploader(store, contributor_token)
        if not 1 <= len(files) <= MAX_FILES_PER_UPLOAD:     # (a form of more files is refused unread: UploadRoute)
            raise _file_count_refused(len(files))
        landmark = get_landmark(landmark_id.strip()) if landmark_id and landmark_id.strip() else None
        outcomes = []
        for f in files:
            try:
                f.file.seek(0)
                # One byte past the limit is enough for ingest to reject the file as too large.
                data = f.file.read(settings.max_upload_bytes + 1)
                outcomes.append(ingest_upload(store, settings, filename=f.filename or "", data=data,
                                              contributor=contributor, landmark=landmark))
            except Exception:
                # One file must not lose the others: earlier ones are stored and still need planning.
                log.exception("storing upload %r failed", f.filename)
                outcomes.append(UploadOutcome(f.filename or "", None, False, [R_SERVER_ERROR], []))
        # Plan once per request, so a batch of photos makes one job rather than one per threshold step.
        touched = dict.fromkeys(o.photo.landmark_id for o in outcomes if o.accepted)
        triggered = [job for lm_id in touched for job in runner.schedule(lm_id)]
        return {"results": [upload_result(o, salt) for o in outcomes],
                "triggered_jobs": [job_info(j) for j in triggered]}

    app.include_router(uploads)

    @app.post(WITHDRAW_ROUTE)
    def withdraw_share_location(photo_id: str, body: ShareLocationIn):
        """The uploader withdraws a photo (D30, D32): its public pose (the poses row) and its photo (GET
        .../poses/{public_id}/photo, which every view asks again) are hidden at once, both gated by
        records.location_shared. Either with the photo's manage token (token: from its upload's response, or from
        OWN_PHOTOS_ROUTE), or with the token of the registration it was uploaded under (contributor_token, D34: a
        manage token can be lost, the photo must stay withdrawable). Exactly one of the two, else a 422; no 422 of
        this route echoes any input (validation_error). Every refusal is the same 403 WITHDRAWAL_REFUSED: a wrong
        manage token whether or not the photo exists; and for the registration, a missing, malformed or unknown
        token, a photo that does not exist, one of another registration and one of none alike, so the answer tells
        nobody whether a photo exists or whose it is. The accepted agreement version is not checked: withdrawing
        must not take accepting a new agreement. Withdrawing again, or a rejected photo of one's own, is a 200:
        it is not shared. 404 only for a right manage token of a photo the store no longer has."""
        if (body.token is None) == (body.contributor_token is None):
            raise _invalid("token", "send the photo's manage token (token) or the contributor token "
                                    f"({TOKEN_FIELD}): one of them", None)
        if body.token is not None:
            if not check_manage_token(photo_id, body.token, salt):
                raise HTTPException(403, WITHDRAWAL_REFUSED)
            try:
                store.withdraw_location_consent(photo_id)
            except NotFound:
                raise HTTPException(404, "unknown photo") from None
        else:
            contributor = contributors.authenticate(store, body.contributor_token)
            try:
                if contributor is None:
                    raise NotFound(photo_id)
                store.withdraw_location_consent(photo_id, contributor.id)
            except NotFound:
                raise HTTPException(403, WITHDRAWAL_REFUSED) from None
        return {"photo_id": photo_id, "share_location": False}

    @app.post("/api/landmarks/{landmark_id}/rebuild", status_code=202)
    def rebuild(landmark_id: str, body: RebuildIn):
        lm = get_landmark(landmark_id)
        sector = None
        if body.kind == JOB_FACE:
            sector = body.sector
            if sector is None or not 0 <= sector < settings.num_sectors:
                raise _invalid("sector", f"face rebuilds need a sector in 0..{settings.num_sectors - 1}", sector)
        try:
            job = runner.submit_manual(lm.id, body.kind, sector)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return job_info(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            job = store.get_job(job_id)
        except NotFound:
            job = None
        if job is None or job.kind == JOB_LOCALIZE:
            # a localize job is internal (D30): its times and photo count would tell when consented photos arrived
            raise HTTPException(404, f"unknown job {_quoted(job_id)}")
        return job_info(job)

    # The two file mounts are all that is ever compressed (D34): no route's JSON, which may hold a token, and not
    # the photo route, whose JPEG is compressed already. One cache of compressed bytes for both, so one bound.
    app.state.gzip_cache = GzipCache()
    app.mount("/files/models", ModelFiles(directory=settings.models_dir, gzip_cache=app.state.gzip_cache),
              name="models")
    # Mounted last so every /api route above takes precedence.
    app.mount("/", WebFiles(directory=WEB_DIR, html=True, gzip_cache=app.state.gzip_cache), name="web")
    app.add_middleware(BodySizeLimit, max_bytes=SMALL_BODY_MAX_BYTES,
                       upload_max_bytes=MAX_FILES_PER_UPLOAD * settings.max_upload_bytes + BODY_SLACK_BYTES)
    return app


def _invalid(field: str, msg: str, value: Any, type: str = "value_error",
             ctx: dict[str, Any] | None = None) -> RequestValidationError:
    """A 422 shaped like FastAPI's own validation errors, so clients parse one format."""
    error = {"type": type, "loc": ("body", field), "msg": msg, "input": value}
    if ctx is not None:
        error["ctx"] = ctx
    return RequestValidationError([error])


def _check_terms(terms: object, doing: str) -> None:
    """Refuse anything but the current user agreement version (D32, D33: at registration and when accepting a new
    version): 422 terms_required (missing, null or blank) or terms_outdated (any other value), at loc terms, input
    null (an acceptance's body holds the token: nothing of a body is echoed); ctx names the current version and the
    agreement's page."""
    if terms is None or (isinstance(terms, str) and not terms.strip()):
        raise _terms_refused("terms_required", doing)
    if terms != TERMS_CURRENT:
        raise _terms_refused("terms_outdated", doing)


def _terms_refused(type: str, doing: str) -> RequestValidationError:
    return _invalid("terms", f"{doing} needs agreeing to the user agreement {TERMS_CURRENT} ({TERMS_URL})", None,
                    type=type, ctx={"version": TERMS_CURRENT, "url": TERMS_URL})


def _register_required(doing: str) -> RequestValidationError:
    """The 422 of a missing, blank, malformed or unknown contributor token (D33): type register_required, input None,
    so the token never comes back in a response."""
    return _invalid(TOKEN_FIELD, f"{doing} needs a registered contributor (POST {REGISTER_ROUTE})", None,
                    type="register_required")


def _uploader(store: Store, token: object) -> Contributor:
    """The registered contributor an upload's token names, who has accepted the current user agreement (D33); else
    the 422 register_required or terms_outdated, both with input None."""
    contributor = contributors.authenticate(store, token)
    if contributor is None:
        raise _register_required("uploading")
    if contributor.terms != TERMS_CURRENT:
        raise _invalid(TOKEN_FIELD, f"the contributor has not accepted the user agreement {TERMS_CURRENT} "
                       f"({TERMS_URL}); accept it with POST {ACCEPT_ROUTE}", None, type="terms_outdated",
                       ctx={"version": TERMS_CURRENT, "url": TERMS_URL})
    return contributor


def _file_count_refused(count: int | None) -> RequestValidationError:
    return _invalid("files", f"upload 1-{MAX_FILES_PER_UPLOAD} files at a time", count)


def _upload_refusal(store: Store, refused: _FormRefused) -> Exception:
    """The answer to an upload whose form the parser refused (D34, UploadRequest). More than MAX_FILES_PER_UPLOAD
    file parts: what such an upload always got, in the same order, from the text fields read before the file over
    the limit: register_required or terms_outdated for the token, else the 422 of the file count. So a client must
    send the token before the files, or a registered contributor's 51 files are answered register_required: this
    project's page does (web/api.js appends the text fields first; a browser sends a form's parts in the order they
    were appended), and the README says so for other clients. More than UPLOAD_MAX_FIELDS text fields, or one over
    UPLOAD_FIELD_MAX_BYTES: 413. Anything else is a malformed form: Starlette's own 400."""
    if refused.detail.startswith("Too many files"):
        token = next((value for name, value in reversed(refused.fields) if name == TOKEN_FIELD), None)
        try:
            _uploader(store, token)
        except RequestValidationError as exc:
            return exc
        return _file_count_refused(None)
    if refused.detail.startswith(("Too many fields", "Part exceeded maximum size")):
        return HTTPException(413, f"upload form over {UPLOAD_MAX_FIELDS} text fields, or a text field over "
                                  f"{UPLOAD_FIELD_MAX_BYTES} bytes")
    return refused


def _rendered_error(error: dict[str, Any]) -> dict[str, Any]:
    """One validation error as a 422 carries it: jsonable_encoder's output, NaN and infinity written as strings
    (_json_float: Python's json accepts the NaN literal), and input null when the input cannot be rendered as the
    response's JSON or is too big to send back. Cannot be rendered: it holds a string that does not encode to UTF-8
    (JSON's \\ud800 escape gives a lone surrogate), anywhere in it, keys too, or is nested too deep to encode; either
    would turn the 422 into a 500. Too big: its JSON is over MAX_ECHO_BYTES (D34)."""
    if _plainly_too_big(error.get("input")):
        error = {**error, "input": None}
    try:
        rendered = jsonable_encoder(error, custom_encoder={float: _json_float})
        json.dumps(rendered, ensure_ascii=False, allow_nan=False).encode("utf-8")      # as JSONResponse renders it
        echo = json.dumps(rendered.get("input"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, RecursionError):        # UnicodeEncodeError is a ValueError
        echo = None
    if echo is None or len(echo) > MAX_ECHO_BYTES:
        rendered = jsonable_encoder({**error, "input": None}, custom_encoder={float: _json_float})
    return rendered


def _plainly_too_big(value: object) -> bool:
    """Whether value, as the JSON data a parsed body is made of, is far over MAX_ECHO_BYTES: one pass of the C
    encoder, before jsonable_encoder walks a body of thousands of items in Python once per error. Twice the cap,
    counted in characters: what _rendered_error measures is never longer than half of this (it writes a non-finite
    number as a shorter string), so nothing it would echo is dropped here. False for anything this cannot tell (a
    value that is not JSON data, or does not render): _rendered_error then decides."""
    try:
        return len(json.dumps(value, ensure_ascii=False)) > 2 * MAX_ECHO_BYTES
    except (TypeError, ValueError, RecursionError):
        return False


def _quoted(value: str) -> str:
    """value as an error text quotes it: its repr, cut at MAX_ECHO_CHARS characters (D34). The ids these texts name
    are short; a longer string is not one, and a form field can be a megabyte."""
    return repr(value if len(value) <= MAX_ECHO_CHARS else value[:MAX_ECHO_CHARS] + "...")


def _is_text(value: object) -> bool:
    """Whether value is a str a response can carry: one that encodes to UTF-8 (JSON's \\ud800 escape gives a lone
    surrogate, which does not, and would turn the 422 into a 500)."""
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _json_float(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)


def _finite(value: Any) -> float | None:
    # Starlette's JSONResponse refuses NaN/inf; reconstructor metadata may contain them.
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def model_url(settings: Settings, model: ModelRecord) -> str:
    rel = (settings.data_dir / model.ply_path).relative_to(settings.models_dir).as_posix()
    return f"/files/models/{rel}"


def stylized_info(settings: Settings, model: ModelRecord, pending: bool = False) -> dict[str, Any] | None:
    """The model's stylized GLB: ready (its URL, whether the photos' registration was accepted, which the client's
    label shows, the landmark's public poses, and snapped: whether the GLB's footprint snap was accepted, which moved
    this model's cameras in the public poses but not its point cloud), pending (a stylize job for it is queued or
    running), or None (none made, none coming)."""
    rec = model.stylized
    if rec and rec.get("glb_path"):
        reg = rec.get("registration") or {}
        accepted = bool(reg.get("accepted"))
        rel = (settings.data_dir / rec["glb_path"]).relative_to(settings.models_dir).as_posix()
        return {"status": "ready", "url": f"/files/models/{rel}",
                "registration": {"accepted": accepted, "reason": public_text(reg.get("reason"), {model.id})},
                "poses_url": poses_url(model.landmark_id),
                "snapped": snap_accepted(recorded_snap(settings, model))}
    if pending:
        return {"status": "pending", "url": None, "registration": None, "poses_url": None, "snapped": None}
    return None


def poses_url(landmark_id: str) -> str:
    """The landmark's public photo poses (GET /api/landmarks/{id}/poses), in the GLB's frame."""
    return f"/api/landmarks/{quote(landmark_id, safe='')}/poses"


def poses_snapshot_time(jobs: list[Job]) -> float | None:
    """When the landmark's newest reconstruction job (facade or full) finished, done or failed; None before any did.
    The public poses cover the photos uploaded by then: they change when a reconstruction job finishes, not with each
    upload, so a new row cannot be told apart from the other uploads of its batch (D30, owner question 2)."""
    return max((j.finished_at for j in jobs if j.kind in (JOB_FACE, JOB_FULL) and j.status in (JOB_DONE, JOB_FAILED)
                and j.finished_at is not None), default=None)


def landmark_poses(store: Store, settings: Settings, lm: Landmark, salt: str) -> dict[str, Any]:
    """The public photo poses of a landmark (D30), format crowd3d.photo-poses version 2: the rows of
    photoposes.public_poses (shared photos only: uploaded under the user agreement and not withdrawn, D32; whole
    metres and degrees, keyed by a salted public id, sorted by
    it; no uploader, time, model or store id) for the accepted photos uploaded by the snapshot time as_of
    (poses_snapshot_time). Consent is read now, so a withdrawal takes effect at once. The visual localizations of
    the localize jobs (source "loc") count as they are stored, and only while Settings.localize_photos is on
    (jobs.shown_localizations: off, every stored one is withheld at once): they are of photos in the snapshot (or
    wait for it like their photos), and a localize job runs when the maps change, right after a reconstruction job,
    or when photos arrive. frame: landmark ENU metres, the stylized GLB's frame (SfM cameras and
    localized ones moved with their model's accepted snap, stylized_snaps); rounding; uncertainty: what sp / sy are and
    how often they held, per source (photoposes.public_uncertainty); counts over the snapshot."""
    as_of = poses_snapshot_time(store.list_jobs(lm.id))
    photos = [p for p in store.list_photos(lm.id, PHOTO_ACCEPTED) if as_of is not None and p.uploaded_at <= as_of]
    models = store.published_models(lm.id)
    shared = [p for p in photos if location_shared(p)]
    body = public_poses(photos, models, lm.lat, lm.lon, salt=salt, snaps=stylized_snaps(settings, models),
                        localizations=shown_localizations(settings, store, lm.id) if shared else None,
                        aspects=processed_aspects(shared, settings.data_dir))
    return {"format": POSES_FORMAT, "version": POSES_VERSION,
            "landmark": {"id": lm.id, "name": lm.name},
            "frame": {"name": "landmark-enu", "origin": {"lat": lm.lat, "lon": lm.lon}, "axes": "x east, y north, z up",
                      "unit": "m"},
            "as_of": as_of, "rounding": public_rounding(), "uncertainty": public_uncertainty(), **body}


def shared_photo(store: Store, settings: Settings, lm: Landmark, salt: str, pid: str) -> PhotoRecord | None:
    """The photo whose public id is pid, if other people may see it now (D32), else None: pid is the id of a row of
    the landmark's public poses (landmark_poses: the snapshot, shared, a determined pose; the one rule, so the photos
    served are exactly the rows shown), and the photo, read again from the store, is still shared
    (records.location_shared)."""
    if not _PUBLIC_ID.fullmatch(pid):
        return None
    if not any(row["id"] == pid for row in landmark_poses(store, settings, lm, salt)["photos"]):
        return None
    return next((p for p in store.list_photos(lm.id, PHOTO_ACCEPTED)
                 if location_shared(p) and hmac.compare_digest(public_id(p.id, salt), pid)), None)


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    """Whether an If-None-Match header names etag (weak comparison, as RFC 9110 13.1.2 has it: W/ is ignored) or is
    "*"."""
    if not if_none_match:
        return False
    tags = [t.strip() for t in if_none_match.split(",")]
    return "*" in tags or etag.removeprefix("W/") in {t.removeprefix("W/") for t in tags}


def geo_info(settings: Settings, meta: dict[str, Any]) -> dict[str, Any] | None:
    """Where the model sits on the map and how sure that is: its GeoAnchor (ARCHITECTURE 3.5.9) with the current
    calibration floors applied. level (full / prior_only / none); sigma, the 1-sigma per component (yaw_deg,
    log_scale, east_m and north_m at the pivot, up_m, tilt_deg); source per component (yaw, scale, xy, up, gravity);
    calibration_id; position_sigma_m, the horizontal and vertical 1-sigma at the points' horizontal median (the error at
    the walls); calibrated, whether that calibration passed the held-out check (Settings.calibrated_anchor_ids: the
    landmark page shows "位置 ±X 米（1σ）" only then). None for a model recorded before anchors existed. Informational:
    the publish gate does not read it."""
    try:
        anchor = read_anchor(meta)
        if anchor is None:
            return None
        anchor = anchor.recalibrated()
        h, v = anchor.position_sigma_m(meta.get("points_median_en") or anchor.pivot_enu)
    except (KeyError, IndexError, TypeError, ValueError):      # a malformed record is reported as none
        return None
    cal_id = anchor.calibration.get("id")
    return {"level": anchor.level,
            "sigma": {k: _finite(v_) for k, v_ in anchor.sigma.items()},
            "source": dict(anchor.source),
            "calibration_id": cal_id,
            "position_sigma_m": {"horizontal": _finite(round(h, 2)), "vertical": _finite(round(v, 2))},
            "calibrated": cal_id is not None and cal_id in settings.calibrated_anchor_ids}


def stylizing(store: Store, landmark_id: str) -> set[str]:
    """Ids of the landmark's models a stylize job is queued or running for."""
    return {j.model_id for j in store.list_jobs(landmark_id, ACTIVE_JOB_STATUSES) if j.kind == JOB_STYLIZE}


def model_info(settings: Settings, model: ModelRecord,
               pending: set[str] | frozenset[str] = frozenset()) -> dict[str, Any]:
    """pending: ids of models whose stylize job is queued or running (stylizing)."""
    meta = model.meta
    return {
        "id": model.id,
        "kind": model.kind,
        "sector": model.sector,
        "url": model_url(settings, model),
        "created_at": model.created_at,
        "num_photos": model.num_photos,
        "num_registered": int(meta.get("num_registered") or 0),
        "num_points": int(meta.get("num_points") or 0),
        "georegistered": bool(meta.get("georegistered")),
        "alignment_rmse_m": _finite(meta.get("alignment_rmse_m")),
        "heading_mae_deg": _finite(meta.get("heading_mae_deg")),
        "geo": geo_info(settings, meta),
        "stylized": stylized_info(settings, model, model.id in pending),
    }


def public_text(text: str | None, keep: frozenset[str] | set[str] = frozenset()) -> str | None:
    """text with every absolute path replaced by <path> and every 32+ hex digit id (store photo ids, content hashes) by
    <id>, except a model id after "model " and the ids in keep (ids the response serves anyway)."""
    if not text:
        return text
    text = _PATH_IN_TEXT.sub("<path>", text)
    return _HEX_ID_IN_TEXT.sub(lambda m: m.group(0) if m.group(0) in keep else "<id>", text)


def job_info(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "landmark_id": job.landmark_id,
        "kind": job.kind,
        "sector": job.sector,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": public_text(job.error, {job.id, job.model_id} - {None}),   # the store keeps the full message
        "num_photos": len(job.photo_ids),
        "model_id": job.model_id,
    }


def camera_estimates(models: list[ModelRecord]) -> dict[str, dict[str, Any]]:
    """photo_id -> camera entry of the newest model containing it; full models win over face models.

    Only published models count, and a camera with flags (its pose disagrees with the photo's own
    GPS/compass, lies outside the radius, ...) is not an estimate: an older model's entry stays.
    Internal only (the benchmark and photoposes read it): the entries are exact poses keyed by store photo id,
    so no endpoint serves them (D30).
    """
    estimates: dict[str, dict[str, Any]] = {}
    published = [m for m in models if m.meta.get("published", True)]
    for m in sorted(published, key=lambda m: (m.kind == JOB_FULL, m.created_at, m.id)):
        for pid, cam in (m.meta.get("cameras") or {}).items():
            if isinstance(cam, dict) and not cam.get("flags"):
                estimates[pid] = cam
    return estimates


def upload_result(outcome: UploadOutcome, salt: str) -> dict[str, Any]:
    """The uploader's own result for one file. share_location: whether other people may see the photo's rounded pose
    and the photo (records.location_shared: true for every accepted photo, as every upload agreed to the user
    agreement, D32); only then public_id, the id its row carries in the public poses (consent.public_id), and
    manage_token, the token to withdraw it (consent.manage_token). Both come again, to the registration it was
    uploaded under, from OWN_PHOTOS_ROUTE (own_photo, D34)."""
    p = outcome.photo
    shared = p is not None and location_shared(p)
    return {
        "filename": outcome.filename,
        "photo_id": p.id if p else None,
        "accepted": outcome.accepted,
        "reasons": outcome.reasons,
        "warnings": outcome.warnings,
        "landmark_id": p.landmark_id if p else None,
        "sector": p.sector if p else None,
        "bearing_deg": p.bearing_deg if p else None,
        "distance_m": p.distance_m if p else None,
        "share_location": shared,
        "public_id": public_id(p.id, salt) if shared else None,
        "manage_token": manage_token(p.id, salt) if shared else None,
    }


def own_photo(photo: PhotoRecord, salt: str) -> dict[str, Any]:
    """One shared photo as its own registration gets it back (OWN_PHOTOS_ROUTE, D34): the ids and the manage token
    its upload's result carried (upload_result), nothing else of the photo. The file name is not among them: the
    store never kept it."""
    return {"photo_id": photo.id, "landmark_id": photo.landmark_id,
            "public_id": public_id(photo.id, salt), "manage_token": manage_token(photo.id, salt)}


def landmark_summary(
    store: Store,
    settings: Settings,
    lm: Landmark,
    photos: list[PhotoRecord] | None = None,
    latest: list[ModelRecord] | None = None,
) -> dict[str, Any]:
    photos = store.list_photos(lm.id, PHOTO_ACCEPTED) if photos is None else photos
    latest = store.latest_models(lm.id) if latest is None else latest
    pending = stylizing(store, lm.id)
    cov = compute_coverage(photos, settings)
    face_sectors = {m.sector for m in latest if m.kind == JOB_FACE}
    full = next((m for m in latest if m.kind == JOB_FULL), None)
    return {
        "id": lm.id,
        "name": lm.name,
        "lat": lm.lat,
        "lon": lm.lon,
        "radius_m": lm.radius_m,
        "num_photos": len(photos),
        "num_contributors": len({p.user_id for p in photos}),
        "progress": cov.progress,
        "full_ready": cov.full_ready,
        "sectors": [
            {
                "index": s.index,
                "label": s.label,
                "count": s.count,
                "state": s.state,
                "bearing_center_deg": s.bearing_center_deg,
                "has_face_model": s.index in face_sectors,
            }
            for s in cov.sectors
        ],
        "latest_full_model": model_info(settings, full, pending) if full else None,
    }


def landmark_detail(store: Store, settings: Settings, lm: Landmark) -> dict[str, Any]:
    photos = store.list_photos(lm.id, PHOTO_ACCEPTED)
    latest = store.latest_models(lm.id)
    pending = stylizing(store, lm.id)
    # No per-photo rows (D30): who took which photo where, and the cameras' exact poses, are not public.
    return {
        **landmark_summary(store, settings, lm, photos, latest),
        # nicknames by name, without per-person counts (D30: the counts would tie public poses to nicknames)
        "contributors": [{"user_id": u} for u in store.contributors(lm.id)],
        "models": [model_info(settings, m, pending) for m in latest],
        # Reconstruction jobs only: stylize progress reaches clients through ModelInfo.stylized, and a stylize job in this
        # window would push a facade's failed build out of it (and older pages read every non-full kind as a facade).
        "jobs": [job_info(j) for j in store.list_jobs(lm.id)[::-1] if j.kind in (JOB_FACE, JOB_FULL)][:DETAIL_MAX_JOBS],
        "thresholds": {
            "face_min_photos": settings.face_min_photos,
            "full_min_photos": settings.full_min_photos,
            "full_min_sectors": compute_coverage(photos, settings).full_min_sectors,
            "num_sectors": settings.num_sectors,
        },
    }


def ensure_demo_landmark(store: Store) -> Landmark:
    """The landmark the synthetic demo photos are taken around; created if missing."""
    from . import synthetic  # pulls in OpenCV, so only when asked for

    try:
        return store.get_landmark(DEMO_LANDMARK_ID)
    except NotFound:
        pass
    lm = Landmark(
        id=DEMO_LANDMARK_ID,
        name=synthetic.DEMO_NAME,
        lat=synthetic.DEMO_LAT,
        lon=synthetic.DEMO_LON,
        radius_m=DEMO_RADIUS_M,
        created_at=time.time(),
    )
    try:
        return store.create_landmark(lm)
    except AlreadyExists:
        return store.get_landmark(DEMO_LANDMARK_ID)


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m crowd3d.api [--host 127.0.0.1] [--port 8000] [--data-dir data] [--reconstructor colmap] [--seed-demo]"""
    ap = argparse.ArgumentParser(prog="python -m crowd3d.api", description="Crowd3D HTTP API and web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", type=Path, default=None, help="default: $CROWD3D_DATA_DIR or ./data")
    ap.add_argument(
        "--reconstructor", choices=("colmap", "fake"), default=None, help="default: $CROWD3D_RECONSTRUCTOR or colmap"
    )
    ap.add_argument("--seed-demo", action="store_true", help=f"create the {DEMO_LANDMARK_ID!r} landmark if missing")
    args = ap.parse_args(argv)

    settings = Settings()
    if args.data_dir is not None:
        settings = replace(settings, data_dir=args.data_dir)
    if args.reconstructor is not None:
        settings = replace(settings, reconstructor=args.reconstructor)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(settings)
    if args.seed_demo:
        lm = ensure_demo_landmark(app.state.store)
        logging.getLogger(__name__).info("demo landmark %s (%s) at %.4f, %.4f", lm.id, lm.name, lm.lat, lm.lon)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
