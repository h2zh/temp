// Fetch helpers for the Crowd3D HTTP API (shapes in the prototype spec, "HTTP API"), plus the texts for the codes the
// API returns. The server sends codes and technical English only (D32); the texts are in web/strings.js under api.*
// and read in the page language (web/i18n.js).

import { t, keyed } from './i18n.js';

// Rejection reasons of an upload result (crowd3d/validation.py; server_error: crowd3d/api.py R_SERVER_ERROR). The text
// of not_subject names Settings.subject_max_distance_m, 50 m (tests/test_validation.py checks that they agree).
export const REASON_TEXT = keyed('api.reason', [
  'unreadable_image',
  'no_exif',
  'no_gps',
  'no_landmark',
  'too_far',
  'facing_away',
  'too_small',
  'too_blurry',
  'duplicate',
  'too_large',
  'not_subject', // D31
  'server_error',
]);

// subject_unchecked: the landmark has no bound OSM building outline
export const WARNING_TEXT = keyed('api.warning', ['no_heading', 'magnetic_heading', 'subject_unchecked']);

// The version of the user agreement (web/terms.html, data-terms-version) a contributor accepts when registering, or
// once more after a new version (D32, D33); the server registers and accepts only the current one and refuses uploads
// of a contributor who has not accepted it (crowd3d/records.py TERMS_CURRENT; tests pin that the three agree).
export const TERMS_VERSION = 'terms-v1';

export const JOB_STATUS_TEXT = keyed('api.jobStatus', ['queued', 'running', 'done', 'failed']);

// Job kinds besides the models' own (MODEL_KIND_TEXT): a stylize job makes a published model's stylized GLB.
export const JOB_KIND_TEXT = keyed('api.jobKind', ['stylize']);

export const MODEL_KIND_TEXT = keyed('api.modelKind', ['face', 'full']);

// Job errors are English free text from crowd3d/reconstruct.py and jobs.py (their wording is what these match); the
// known ones get a text of their own, the rest the fallback (the page shows the raw error under its technical-details
// label). The counts stay as written: strings, not formatted numbers.
const JOB_ERROR_TEXT = [
  [/^stylize failed.*timed out after/, () => t('api.jobError.stylizeTimeout')],
  [/^stylize failed/, () => t('api.jobError.stylizeFailed')],
  [/^quality check failed/, () => t('api.jobError.qualityCheck')],
  [/^interrupted \d+ times/, () => t('api.jobError.interrupted')],
  [/no initial image pair/, () => t('api.jobError.noInitialPair')],
  [/only (\d+) of (\d+) photos could be registered/, (m) => t('api.jobError.fewRegistered', { registered: m[1], total: m[2] })],
  [/^need at least \d+ photo/, () => t('api.jobError.tooFewPhotos')],
];

// Pillow on the server has no HEIF decoder, so HEIC/HEIF uploads come back as unreadable_image.
const HEIF_NAME = /\.(heic|heif)$/i;

// The request fields a 422 names (detail[].loc), as the forms call them.
const FIELD_TEXT = keyed('api.field', [
  'name',
  'lat',
  'lon',
  'radius_m',
  'id',
  'nickname',
  'files',
  'landmark_id',
  'kind',
  'sector',
  'terms',
]);

const finite = (x) => typeof x === 'number' && Number.isFinite(x);
// a code of the table itself, not an Object.prototype name ('toString')
const own = (table, key) => typeof key === 'string' && Object.hasOwn(table, key);

// A 422 detail item's text by its FastAPI type: (field label, ctx) -> text, or null when ctx lacks the limit. Any
// other type (and a missing limit) gets api.invalid.value.
const INVALID_TEXT = {
  missing: (field) => t('api.invalid.missing', { field }),
  string_too_long: (field, ctx) => (finite(ctx.max_length) ? t('api.invalid.tooLong', { field, n: ctx.max_length }) : null),
  string_too_short: (field, ctx) => (finite(ctx.min_length) ? t('api.invalid.tooShort', { field, n: ctx.min_length }) : null),
  greater_than_equal: (field, ctx) => (finite(ctx.ge) ? t('api.invalid.min', { field, limit: ctx.ge }) : null),
  less_than_equal: (field, ctx) => (finite(ctx.le) ? t('api.invalid.max', { field, limit: ctx.le }) : null),
  greater_than: (field, ctx) => (finite(ctx.gt) ? t('api.invalid.above', { field, limit: ctx.gt }) : null),
};

// The 422 types of a request without a registration the server knows or without the current user agreement (D32,
// D33): a text of their own, without the field.
const TERMS_TEXT = {
  register_required: () => t('api.register.required'),
  terms_required: () => t('api.terms.required'),
  terms_outdated: () => t('api.terms.outdated'),
};

/** The text for a rejection reason or warning code; unknown codes are shown as they are. */
export function codeText(code) {
  if (own(REASON_TEXT, code)) return REASON_TEXT[code];
  if (own(WARNING_TEXT, code)) return WARNING_TEXT[code];
  return String(code);
}

/** Texts of codes (reasons or warnings) as one line: '缺少GPS定位、距离地标太远' / 'No GPS location. Too far from the landmark'. */
export function codeListText(codes) {
  return joinReasons(codes.map(codeText));
}

function joinReasons(texts) {
  return texts.join(t('api.sep.reasons'));
}

/** The text for why a job failed (its raw error is technical: the page shows it under its own label). */
export function jobErrorText(error) {
  const message = String(error ?? '');
  for (const [pattern, text] of JOB_ERROR_TEXT) {
    const m = message.match(pattern);
    if (m) return text(m);
  }
  return t('api.jobError.other');
}

/** The reasons of one UploadResult as one line, with a format hint for HEIC/HEIF files the server cannot decode. */
export function uploadReasonText(result) {
  return joinReasons(
    result.reasons.map((code) =>
      code === 'unreadable_image' && HEIF_NAME.test(result.filename ?? '') ? t('api.reason.heic') : codeText(code),
    ),
  );
}

export function isActiveJob(job) {
  return job.status === 'queued' || job.status === 'running';
}

/** An HTTP error: message is the text for the page (errorMessage), body the server's answer as it came (technical). */
export class ApiError extends Error {
  constructor(status, message, body = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }

  /** The server's own detail ({detail: str} or FastAPI's [{type, loc, msg, ...}]): technical English, for debugging. */
  get detail() {
    return this.body?.detail ?? null;
  }
}

/**
 * The type of a refused request's first 422 item (register_required, terms_outdated, ...), for the pages that act on
 * it (D33); null for any other error.
 */
export function refusalType(err) {
  if (!(err instanceof ApiError) || err.status !== 422 || !Array.isArray(err.detail)) return null;
  const type = err.detail[0]?.type;
  return typeof type === 'string' ? type : null;
}

/** The text of one 422 detail item: the agreement's own text, or the field's label with a text by type. */
function invalidText(d) {
  if (own(TERMS_TEXT, d?.type)) return TERMS_TEXT[d.type]();
  const name = Array.isArray(d?.loc) ? [...d.loc].reverse().find((x) => typeof x === 'string') : undefined;
  if (!name || name === 'body') return t('api.invalid.request');
  const field = own(FIELD_TEXT, name) ? FIELD_TEXT[name] : name; // an unknown field: its code, as the server names it
  const ctx = d.ctx && typeof d.ctx === 'object' ? d.ctx : {};
  return (own(INVALID_TEXT, d.type) ? INVALID_TEXT[d.type](field, ctx) : null) ?? t('api.invalid.value', { field });
}

/**
 * The page's text for a failed request: status 0 (no answer), the statuses the API uses with a text each, and for a
 * 422 the text of each of its detail items (FastAPI's [{type, loc, msg, ctx}]). The server's own detail and msg are
 * technical English and never shown (ApiError.detail keeps them).
 */
export function errorMessage(status, body) {
  const detail = body?.detail;
  if (status === 422 && Array.isArray(detail) && detail.length) {
    return detail.map(invalidText).join(t('api.sep.errors'));
  }
  if (status === 0) return t('api.unreachable');
  if (status === 403) return t('api.status.403');
  if (status === 404) return t('api.status.404');
  if (status === 409) return t('api.status.409');
  if (status === 413) return t('api.status.413');
  if (status >= 500 && status <= 599) return t('api.status.server', { status: String(status) });
  return t('api.status.other', { status: String(status) });
}

async function request(path, { method = 'GET', json, body } = {}) {
  const init = { method, headers: { Accept: 'application/json' } };
  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body;
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, errorMessage(0, null));
  }
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      // Proxies and crashed servers answer with HTML; fall through to the status message.
    }
  }
  if (!res.ok) throw new ApiError(res.status, errorMessage(res.status, data), data);
  return data;
}

const enc = encodeURIComponent;

export async function listLandmarks() {
  return (await request('/api/landmarks')).landmarks;
}

export function getLandmark(id) {
  return request(`/api/landmarks/${enc(id)}`);
}

/**
 * The landmark's public photo poses (crowd3d.photo-poses v2, D30): the photos uploaded under the user agreement and
 * not withdrawn (D32), rounded by the server; refreshed when a reconstruction job finishes.
 */
export function getPoses(id) {
  return request(`/api/landmarks/${enc(id)}/poses`);
}

/** payload: {name, lat, lon, radius_m?, id?} -> LandmarkSummary */
export function createLandmark(payload) {
  return request('/api/landmarks', { method: 'POST', json: payload });
}

// D33: the secret contributor token travels only in POST bodies (JSON or a form field), never in a URL or a header,
// so it stays out of server and proxy logs.

/**
 * Register a contributor (D33): -> {nickname, terms, contributor_token}. Sends terms=TERMS_VERSION: the caller
 * registers only after the contributor ticked the agreement box. The token comes back this once; the page keeps it.
 */
export function register({ nickname }) {
  return request('/api/contributors', { method: 'POST', json: { nickname, terms: TERMS_VERSION } });
}

/** A registered contributor accepts the current user agreement once, after a new version (D33): -> {nickname, terms}. */
export function acceptTerms(token) {
  return request('/api/contributors/accept', { method: 'POST', json: { contributor_token: token, terms: TERMS_VERSION } });
}

/**
 * -> {results: [UploadResult], triggered_jobs: [JobInfo]}. token: the registered contributor's (D33); the server takes
 * the nickname and the agreement from the registration, so the upload sends neither. The text fields go before the
 * files (D34): a browser sends a form's parts in the order they were appended, and the server stops reading at the
 * 51st file, so a token after the files would not be read and the answer would be register_required.
 */
export function uploadPhotos({ files, token, landmarkId = null }) {
  const form = new FormData();
  if (token) form.append('contributor_token', token);
  if (landmarkId) form.append('landmark_id', landmarkId);
  for (const file of files) form.append('files', file, file.name);
  return request('/api/photos', { method: 'POST', body: form });
}

/**
 * The photos a registration shares now (D34): -> {photos: [{photo_id, landmark_id, public_id, manage_token}]}, at
 * landmarkId only when given. token: the registered contributor's, in the POST body like everywhere. The server hands
 * the registration its photos' manage tokens again, so a token lost with a dropped upload response, another tab or
 * cleared site data comes back as long as this device is registered. Works whatever agreement version was accepted.
 */
export function ownPhotos(token, landmarkId = null) {
  const json = landmarkId ? { contributor_token: token, landmark_id: landmarkId } : { contributor_token: token };
  return request('/api/contributors/photos', { method: 'POST', json });
}

/**
 * Withdraw a photo, with the manage token its upload returned or ownPhotos gave again (D30, D32): hides its public pose
 * and its photo.
 */
export function withdrawShareLocation(photoId, token) {
  return request(`/api/photos/${enc(photoId)}/share_location`, { method: 'POST', json: { share: false, token } });
}

/** kind: "full" | "face" (face needs sector) -> JobInfo */
export function rebuild(landmarkId, kind, sector = null) {
  const json = sector === null || sector === undefined ? { kind } : { kind, sector };
  return request(`/api/landmarks/${enc(landmarkId)}/rebuild`, { method: 'POST', json });
}

export function getJob(jobId) {
  return request(`/api/jobs/${enc(jobId)}`);
}
