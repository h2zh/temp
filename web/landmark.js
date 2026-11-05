// Landmark page: coverage compass, photo upload, contributors, jobs, models and the 3D viewer. Per photo it shows only
// the public poses (D30, GET /api/landmarks/{id}/poses): the photos uploaded under the user agreement and not
// withdrawn (D32), rounded by the server to 1 m and 1 degree (fields of view to 5 degrees), without uploader or time,
// as a table and as markers in the 3D viewer. Uploading needs ticking the agreement box (web/terms.html), which starts
// unticked on every page load and stays ticked for further uploads in the same page view; every upload sends the
// agreement's version (api.js TERMS_VERSION). The manage tokens (and public ids) of the uploaded photos stay in this
// browser (localStorage) so the uploader can find and withdraw them. Texts in the page language (web/strings.js).

import {
  ApiError,
  JOB_KIND_TEXT,
  JOB_STATUS_TEXT,
  MODEL_KIND_TEXT,
  codeListText,
  getJob,
  getLandmark,
  getPoses,
  isActiveJob,
  jobErrorText,
  rebuild,
  uploadPhotos,
  uploadReasonText,
  withdrawShareLocation,
} from './api.js';
import { compassSvg, sectorFace, sectorTitle } from './compass.js';
import { sectorSideBearing } from './framing.js';
import { gpsResidualLabel, modelToShow, modelsKey, positionSigmaM, sortModels, stylizedView } from './models.js';
import {
  SOURCE_TEXT,
  cloudMarkers,
  countsText,
  coverageText,
  headingText,
  markersKey,
  parsePoses,
  positionText,
  rangeText,
  snappedCloudNote,
  sourcesText,
} from './poses.js';
import {
  FULL_REFRESH_MS,
  POLL_RUNNING_MS,
  failedFaceSectors,
  isLatestFailure,
  jobsChanged,
  nextStep,
  pollDelay,
  retryText,
} from './status.js';
import { applyDom, fmtList, t } from './i18n.js';
import { el, fill, fmtMeters, fmtPercent, fmtTime, mountPrefs, progressBar } from './ui.js';

applyDom();
mountPrefs(document.getElementById('prefs'));

const MAX_FILES = 50;
const USER_KEY = 'crowd3d.user_id';
const SHARED_KEY_PREFIX = 'crowd3d.shared_location.'; // + landmark id: [{photo_id, token, filename, public_id}]

const $ = (id) => document.getElementById(id);
const landmarkId = new URLSearchParams(location.search).get('id');

const state = {
  detail: null,
  detailAt: 0, // Date.now() of the last full details fetch
  pollTimer: null,
  refreshSeq: 0,
  viewer: null,
  viewerPromise: null,
  shownModel: null,
  userPickedModel: false,
  modelsKey: '',
  rebuildKey: '',
  shared: [], // this browser's shared photos here, with their manage tokens (kept in memory when storage is blocked)
  poses: null, // the public poses document (parsePoses), fetched with every full refresh; null when unavailable
  markersKey: '', // the markers on the viewer now (markersKey)
};

function readUserId() {
  try {
    return localStorage.getItem(USER_KEY) || '';
  } catch {
    return '';
  }
}

function saveUserId(value) {
  try {
    localStorage.setItem(USER_KEY, value);
  } catch {
    // Private mode or blocked storage: the nickname just isn't remembered.
  }
}

function readShared() {
  try {
    const list = JSON.parse(localStorage.getItem(SHARED_KEY_PREFIX + landmarkId) || '[]');
    return Array.isArray(list)
      ? list.filter((s) => s && typeof s.photo_id === 'string' && typeof s.token === 'string')
      : [];
  } catch {
    return [];
  }
}

function saveShared(list) {
  try {
    if (list.length) localStorage.setItem(SHARED_KEY_PREFIX + landmarkId, JSON.stringify(list));
    else localStorage.removeItem(SHARED_KEY_PREFIX + landmarkId);
  } catch {
    // Private mode or blocked storage: the tokens last as long as this page.
  }
}

// A sector by its index, at the start of a line: its name, or "方向 3" / "Side 3" when the details do not list it.
function sectorLabel(d, index) {
  const s = d.sectors[index];
  return s ? sectorTitle(s.label) : t('landmark.sectorFallback', { n: String(index) });
}

function buildTitle(d, kind, sector) {
  if (kind === 'full') return MODEL_KIND_TEXT.full;
  const s = d.sectors[sector];
  return s ? sectorFace(s.label) : sectorFace(null, sectorLabel(d, sector));
}

// A stylize job works on a published model: the full model (no sector) or a facade's.
function jobTitle(d, j) {
  if (j.kind !== 'stylize') return buildTitle(d, j.kind, j.sector);
  const model = j.sector === null || j.sector === undefined ? 'full' : 'face';
  return t('jobs.stylizeTitle', { title: buildTitle(d, model, j.sector), kind: JOB_KIND_TEXT.stylize });
}

// ---------- hero ----------

function renderHero(d) {
  document.title = t('landmark.docTitle', { name: d.name });
  $('lm-name').textContent = d.name;
  $('lm-meta').textContent = [
    `${d.lat.toFixed(5)}, ${d.lon.toFixed(5)}`,
    t('landmark.radius', { n: Math.round(d.radius_m) }),
    t('common.photoCount', { n: d.num_photos ?? 0 }),
    t('common.contributorCount', { n: d.num_contributors ?? 0 }),
  ].join(' · ');
  fill($('lm-progress'), progressBar(d.progress, t('common.coverage')));
  $('lm-progress-text').textContent = t('common.coverageValue', { percent: fmtPercent(d.progress) });
  $('lm-next').textContent = nextStep(d);
}

// ---------- coverage ----------

function renderCoverage(d) {
  const th = d.thresholds;
  const failed = failedFaceSectors(d.jobs);
  $('compass').innerHTML = compassSvg(d.sectors, { size: 260, faceMin: th.face_min_photos, failed });
  $('legend-failed').hidden = !failed.size;
  $('coverage-rule').textContent = t('landmark.coverageRule', {
    sectors: th.num_sectors,
    faceMin: th.face_min_photos,
    fullSectors: th.full_min_sectors,
    fullPhotos: th.full_min_photos,
  });
}

// ---------- contributors ----------

// Nicknames only, by name: no per-person photo counts or ranks (D30), which would tie public photo poses to people.
function renderContributors(d) {
  const list = [...d.contributors].sort((a, b) => a.user_id.localeCompare(b.user_id));
  if (!list.length) {
    fill($('contributors'), el('li', { class: 'empty', text: t('contributors.empty') }));
    return;
  }
  const me = readUserId();
  fill(
    $('contributors'),
    list.map((c) =>
      el(
        'li',
        { class: c.user_id === me ? 'contributor is-me' : 'contributor' },
        c.user_id === me ? t('contributors.me', { name: c.user_id }) : c.user_id,
      ),
    ),
  );
}

// ---------- public photo poses ----------

function poseRows() {
  return state.poses?.photos ?? [];
}

// The table of public poses: source, position from the centre, heading, 2-sigma range; "yours" for this browser's own
// shared photos (their public ids came back with the upload).
function renderPoses() {
  const doc = state.poses;
  const rows = poseRows();
  $('poses-card').hidden = !doc;
  renderMarkerControls();
  if (!doc) return;
  $('poses-as-of').textContent = doc.as_of ? t('poses.asOf', { time: fmtTime(doc.as_of) }) : '';
  $('poses-count').textContent = doc.as_of ? t('common.sentence', { text: countsText(doc.counts) }) : t('poses.noBuildYet');
  $('pose-table').hidden = !rows.length;
  const mine = new Set(state.shared.map((s) => s.public_id).filter(Boolean));
  fill(
    $('pose-rows'),
    rows.map((r) =>
      el(
        'tr',
        {},
        el(
          'td',
          {},
          el('span', { text: SOURCE_TEXT[r.src] }),
          mine.has(r.id) ? el('span', { class: 'chip chip-mine', text: t('poses.mine') }) : null,
        ),
        el('td', { text: positionText(r.p) }),
        el('td', { text: headingText(r.yaw) }),
        el('td', { text: rangeText(r) }),
      ),
    ),
  );
  $('poses-sources').textContent = sourcesText(rows);
  $('poses-coverage').textContent = coverageText(doc.uncertainty, rows);
}

// The model on screen as the latest details describe it (its stylized version may have changed since it was shown).
function shownModelNow() {
  const id = state.shownModel?.id;
  return state.detail?.models.find((m) => m.id === id) ?? state.shownModel;
}

// The markers toggle, or why the markers are not on this model's point cloud (cloudMarkers: a snapped model).
function renderMarkerControls() {
  const shown = poseRows().length > 0;
  const snapped = shownModelNow()?.stylized?.snapped === true;
  $('show-poses-label').hidden = !shown || snapped;
  const note = $('poses-snap-note');
  note.textContent = shown && snapped ? snappedCloudNote() : '';
  note.hidden = !shown || !snapped;
}

function cloudMarkersKey(rows, model) {
  return `${model?.stylized?.snapped === true}|${markersKey(rows)}`;
}

// The public poses as markers on the model in the viewer, when they changed since they were drawn.
function updateMarkers() {
  const model = shownModelNow();
  const key = cloudMarkersKey(poseRows(), model);
  if (!state.viewer || !state.shownModel || key === state.markersKey) return;
  state.markersKey = key;
  state.viewer.setCameras(cloudMarkers(poseRows(), model));
}

// ---------- jobs ----------

function jobDetail(j) {
  if (j.status === 'queued') return t('jobs.queuedAt', { time: fmtTime(j.created_at) });
  if (j.status === 'running') return t('jobs.startedAt', { time: fmtTime(j.started_at) });
  if (j.status === 'done') {
    const secs = j.started_at && j.finished_at ? Math.max(1, Math.round(j.finished_at - j.started_at)) : null;
    const time = fmtTime(j.finished_at);
    return secs ? t('jobs.doneAtTook', { time, n: secs }) : t('jobs.doneAt', { time });
  }
  return t('jobs.failedAt', { time: fmtTime(j.finished_at || j.created_at) });
}

function renderJobs(d) {
  if (!d.jobs.length) {
    fill($('job-list'), el('li', { class: 'empty', text: t('jobs.empty') }));
    return;
  }
  fill(
    $('job-list'),
    d.jobs.map((j) =>
      el(
        'li',
        { class: 'job' },
        el(
          'div',
          { class: 'job-main' },
          el('span', { class: 'job-title', text: jobTitle(d, j) }),
          el('span', { class: `chip chip-${j.status}`, text: JOB_STATUS_TEXT[j.status] ?? j.status }),
        ),
        el('div', { class: 'job-meta muted', text: `${t('common.photoCount', { n: j.num_photos })} · ${jobDetail(j)}` }),
        j.status === 'failed' ? jobFailure(d, j) : null,
      ),
    ),
  );
}

// Why it failed in the page language, how it gets retried (newest attempt only), and the raw message (technical
// English) for whoever debugs it.
function jobFailure(d, j) {
  return [
    el('div', { class: 'job-error', text: jobErrorText(j.error) }),
    isLatestFailure(d.jobs, j) ? el('div', { class: 'job-retry muted', text: t('common.sentence', { text: retryText(d, j) }) }) : null,
    j.error
      ? el('details', { class: 'job-error-detail muted' }, el('summary', { text: t('jobs.technical') }), el('code', { text: j.error }))
      : null,
  ];
}

// ---------- models + viewer ----------

function modelMeta(m) {
  const parts = [
    t('models.points', { n: m.num_points ?? 0 }),
    t('models.registered', { registered: m.num_registered, n: m.num_photos }),
  ];
  if (m.georegistered) {
    const residual = m.alignment_rmse_m !== null && m.alignment_rmse_m !== undefined;
    parts.push(
      residual
        ? t('models.geoResidual', { label: gpsResidualLabel(), value: fmtMeters(m.alignment_rmse_m) })
        : t('models.geo'),
    );
    const sigma = positionSigmaM(m);
    if (sigma !== null) parts.push(t('models.positionSigma', { value: fmtMeters(sigma) }));
  } else {
    parts.push(t('models.notGeo'));
  }
  if (m.heading_mae_deg !== null && m.heading_mae_deg !== undefined) {
    parts.push(t('models.headingError', { value: m.heading_mae_deg.toFixed(1) }));
  }
  return parts.join(' · ');
}

function renderModels(d) {
  const models = sortModels(d.models);
  // Polling re-renders every 3 s; only rebuild the buttons when the models (or their stylized versions) changed so
  // keyboard focus survives.
  const key = modelsKey(models);
  if (key !== state.modelsKey) {
    state.modelsKey = key;
    fillModelList(d, models);
  }
  if (!models.length) return;

  const next = modelToShow(models, state.shownModel, state.userPickedModel);
  if (next) showModel(next);
}

function fillModelList(d, models) {
  const shownId = state.shownModel?.id;
  fill(
    $('model-list'),
    models.map((m) =>
      el(
        'li',
        {},
        el(
          'button',
          {
            type: 'button',
            class: 'model-item',
            'aria-pressed': m.id === shownId ? 'true' : 'false',
            onclick: () => {
              state.userPickedModel = true;
              showModel(m);
            },
          },
          el(
            'span',
            { class: 'model-title' },
            el('span', { class: `kind kind-${m.kind}`, text: buildTitle(d, m.kind, m.sector) }),
            el('span', { class: 'muted', text: fmtTime(m.created_at) }),
          ),
          el('span', { class: 'model-meta muted', text: modelMeta(m) }),
        ),
        stylizedItem(m),
      ),
    ),
  );
}

// The stylized version beside the model's button (a link inside a button would not be clickable on its own).
function stylizedItem(m) {
  const view = stylizedView(m);
  if (!view) return null;
  if (view.pending) return el('span', { class: 'model-stylized muted', text: view.text });
  return el('a', {
    class: `model-stylized${view.unaligned ? ' is-unaligned' : ''}`,
    href: view.href,
    target: '_blank',
    rel: 'noopener',
    text: view.text,
  });
}

function setViewerMessage(text) {
  const msg = $('viewer-message');
  msg.textContent = text || '';
  msg.hidden = !text;
}

function ensureViewer() {
  if (!state.viewerPromise) {
    state.viewerPromise = import('./viewer.js')
      .then(({ ModelViewer }) => {
        state.viewer = new ModelViewer($('viewer'));
        return state.viewer;
      })
      .catch((err) => {
        console.warn('3D viewer unavailable:', err);
        return null;
      });
  }
  return state.viewerPromise;
}

async function showModel(m) {
  state.shownModel = m;
  for (const btn of $('model-list').querySelectorAll('.model-item')) btn.setAttribute('aria-pressed', 'false');
  const index = sortModels(state.detail.models).findIndex((x) => x.id === m.id);
  $('model-list').querySelectorAll('.model-item')[index]?.setAttribute('aria-pressed', 'true');
  $('viewer-caption').textContent = buildTitle(state.detail, m.kind, m.sector);
  renderMarkerControls();

  setViewerMessage(t('viewer.loading'));
  const viewer = await ensureViewer();
  if (state.shownModel !== m) return;
  if (!viewer) {
    setViewerMessage(t('viewer.unavailable'));
    return;
  }
  // Seen from where the photos were taken: a facade from its sector, a full model from the side the sector counts
  // cluster on, or from the viewer's default when they surround it.
  const sector = m.kind === 'face' ? state.detail.sectors[m.sector] : null;
  const viewFromDeg = sector ? sector.bearing_center_deg : sectorSideBearing(state.detail.sectors);
  // the public poses as markers (the first view takes them in while they are shown)
  viewer.setCamerasVisible($('show-poses').checked !== false);
  const rows = poseRows();
  try {
    const cameras = cloudMarkers(rows, m);
    const result = await viewer.load(m.url, viewFromDeg === null ? { cameras } : { viewFromDeg, cameras });
    if (!result) return; // superseded by a newer click
    state.markersKey = cloudMarkersKey(rows, m);
    setViewerMessage('');
  } catch (err) {
    setViewerMessage(t('viewer.loadFailed', { message: err.message || String(err) }));
  }
}

// ---------- manual rebuild ----------

function renderRebuild(d) {
  const targets = [];
  if (d.full_ready) targets.push(['full', MODEL_KIND_TEXT.full]);
  for (const s of d.sectors) {
    if (s.state === 'face_ready') targets.push([`face:${s.index}`, sectorFace(s.label)]);
  }
  const key = targets.map((target) => target[0]).join();
  if (key === state.rebuildKey && $('rebuild-target').options.length) return; // keep an open dropdown open while polling
  state.rebuildKey = key;

  const select = $('rebuild-target');
  const previous = select.value;
  const options = targets.map(([value, text]) => el('option', { value, text }));
  if (!options.length) options.push(el('option', { value: '', text: t('rebuild.none') }));
  fill(select, options);
  if (targets.some((target) => target[0] === previous)) select.value = previous;
  select.disabled = !targets.length;
  $('rebuild-btn').disabled = !targets.length;
}

async function onRebuild() {
  const value = $('rebuild-target').value;
  if (!value) return;
  const [kind, sector] = value.split(':');
  const status = $('rebuild-status');
  const button = $('rebuild-btn');
  button.disabled = true;
  status.className = 'form-status';
  status.textContent = t('rebuild.submitting');
  try {
    const job = await rebuild(landmarkId, kind, sector === undefined ? null : Number(sector));
    status.className = 'form-status is-ok';
    status.textContent = t('rebuild.queued', { title: buildTitle(state.detail, job.kind, job.sector) });
    await refresh();
  } catch (err) {
    status.className = 'form-status is-error';
    status.textContent =
      err instanceof ApiError && err.status === 409 ? t('rebuild.notEnough') : t('rebuild.failed', { message: err.message });
  } finally {
    button.disabled = !$('rebuild-target').value;
  }
}

// ---------- upload ----------

function setUploadStatus(text, kind = '') {
  const status = $('upload-status');
  status.className = `form-status ${kind}`;
  status.textContent = text;
}

function renderUploadResults(res) {
  const d = state.detail;
  const ok = res.results.filter((r) => r.accepted).length;
  const bad = res.results.length - ok;
  let summary = bad ? t('upload.summaryRejected', { n: ok, rejected: bad }) : t('upload.summary', { n: ok });
  if (res.triggered_jobs.length) {
    summary += ` ${t('upload.triggered', { list: fmtList(res.triggered_jobs.map((j) => buildTitle(d, j.kind, j.sector))) })}`;
  }
  setUploadStatus(summary, bad && !ok ? 'is-error' : 'is-ok');

  // Two columns so the reasons get room in the narrow side column: photo (+ where it was taken), result (+ why).
  const rows = res.results.map((r) => {
    let where = null;
    if (r.sector !== null && r.sector !== undefined && d) {
      const side = sectorLabel(d, r.sector);
      where = r.distance_m !== null && r.distance_m !== undefined ? `${side} · ${fmtMeters(r.distance_m)}` : side;
    }
    return el(
      'tr',
      {},
      el(
        'td',
        { class: 'cell-file' },
        el('span', { text: r.filename, title: r.filename }),
        where ? el('span', { class: 'muted', text: where }) : null,
      ),
      el(
        'td',
        { class: 'cell-result' },
        el('span', { class: `chip ${r.accepted ? 'chip-done' : 'chip-failed'}`, text: r.accepted ? t('upload.accepted') : t('upload.rejected') }),
        r.accepted ? el('div', { class: 'muted share-state', text: r.share_location ? t('upload.public') : t('upload.notPublic') }) : null,
        r.reasons.length ? el('div', { text: uploadReasonText(r) }) : null,
        r.warnings.length ? el('div', { class: 'muted', text: t('upload.warnings', { list: codeListText(r.warnings) }) }) : null,
      ),
    );
  });
  fill(
    $('upload-results'),
    el(
      'div',
      { class: 'table-wrap' },
      el(
        'table',
        { class: 'table', 'aria-label': t('upload.results') },
        el(
          'thead',
          {},
          el('tr', {}, [t('upload.col.photo'), t('upload.col.result')].map((h) => el('th', { scope: 'col', text: h }))),
        ),
        el('tbody', {}, rows),
      ),
    ),
  );
  rememberShared(res.results);
}

// ---------- shared photos (withdrawal: hides the photo and its pose) ----------

function rememberShared(results) {
  const known = new Set(state.shared.map((s) => s.photo_id));
  const added = results
    .filter((r) => r.share_location && r.manage_token && r.photo_id && !known.has(r.photo_id))
    .map((r) => ({ photo_id: r.photo_id, token: r.manage_token, filename: r.filename, public_id: r.public_id ?? null }));
  if (!added.length) return;
  state.shared = [...state.shared, ...added];
  saveShared(state.shared);
  renderShared();
}

function renderShared(message = '') {
  const box = $('shared-own');
  const list = state.shared;
  box.hidden = !list.length && !message;
  const status = message ? el('p', { class: 'form-status', role: 'status', text: message }) : null;
  if (!list.length) {
    fill(box, status);
    return;
  }
  fill(
    box,
    el('p', { class: 'shared-own-text', text: t('shared.text', { n: list.length }) }),
    el(
      'ul',
      { class: 'shared-list', 'aria-label': t('shared.listLabel') },
      list.map((s) =>
        el(
          'li',
          { class: 'shared-item' },
          el('span', { class: 'shared-name', text: s.filename || t('shared.unnamed'), title: s.filename || '' }),
          el('button', { type: 'button', class: 'btn btn-ghost btn-small', text: t('shared.withdraw'), onclick: () => withdraw([s]) }),
        ),
      ),
    ),
    list.length > 1
      ? el('button', { type: 'button', class: 'btn btn-ghost', text: t('shared.withdrawAll'), onclick: () => withdraw([...list]) })
      : null,
    status,
  );
}

async function withdraw(items) {
  for (const b of $('shared-own').querySelectorAll('button')) b.disabled = true;
  const done = new Set();
  let failed = 0;
  let lost = 0;
  let refused = 0;
  for (const s of items) {
    try {
      await withdrawShareLocation(s.photo_id, s.token);
      done.add(s.photo_id);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        // the token is right but the server no longer has the photo
        done.add(s.photo_id);
        lost++;
      } else if (err instanceof ApiError && err.status === 403) {
        // the server refuses the token: the photo may well still be shared, so it stays on the list
        refused++;
      } else {
        failed++;
      }
    }
  }
  state.shared = state.shared.filter((s) => !done.has(s.photo_id));
  saveShared(state.shared);
  const ok = done.size - lost;
  const parts = [];
  if (ok) parts.push(t('shared.withdrawn', { n: ok }));
  if (lost) parts.push(t('shared.lost', { n: lost }));
  if (refused) parts.push(t('shared.refused', { n: refused }));
  if (failed) parts.push(t('shared.failed', { n: failed }));
  renderShared(t('common.sentence', { text: parts.join(t('shared.sep')) }));
  if (ok && state.detail) refresh(); // a withdrawn photo leaves the public poses at once
}

async function onUpload(event) {
  event.preventDefault();
  const userInput = $('user-id');
  const filesInput = $('files');
  const userId = userInput.value.trim();
  if (!userId || /[\u0000-\u001f\u007f]/.test(userId)) {
    setUploadStatus(t('upload.nicknameMissing'), 'is-error');
    userInput.focus();
    return;
  }
  const files = [...filesInput.files];
  if (!files.length) {
    setUploadStatus(t('upload.filesMissing'), 'is-error');
    filesInput.focus();
    return;
  }
  if (files.length > MAX_FILES) {
    setUploadStatus(t('upload.tooMany', { max: MAX_FILES, n: files.length }), 'is-error');
    return;
  }
  const agree = $('agree-terms');
  if (agree.checked !== true) {
    // D32: no upload without the user agreement (the server refuses one without it too)
    setUploadStatus(t('api.terms.required'), 'is-error');
    agree.focus();
    return;
  }
  saveUserId(userId);

  const form = $('upload-form');
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  button.textContent = t('upload.submitting');
  form.setAttribute('aria-busy', 'true');
  setUploadStatus(t('upload.checking', { n: files.length }));
  try {
    const res = await uploadPhotos({ files, userId, landmarkId });
    renderUploadResults(res);
    filesInput.value = '';
    await refresh();
  } catch (err) {
    setUploadStatus(t('upload.failed', { message: err.message }), 'is-error');
  } finally {
    button.disabled = false;
    button.textContent = t('upload.submit');
    form.removeAttribute('aria-busy');
  }
}

// ---------- load + poll ----------

function render(d) {
  renderHero(d);
  renderCoverage(d);
  renderContributors(d);
  renderPoses();
  renderJobs(d);
  renderModels(d);
  renderRebuild(d);
  updateMarkers();
}

function showPageError(text) {
  const box = $('page-error');
  box.textContent = text;
  box.hidden = !text;
}

function showFatal(text) {
  $('lm-name').textContent = t('landmark.cannotOpen');
  $('landmark-content').hidden = true;
  fill($('page-error'), el('span', { text }), ' ', el('a', { href: './', text: t('landmark.backToList') }));
  $('page-error').hidden = false;
}

function schedulePoll(d) {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
  const delay = pollDelay(d.jobs);
  $('poll-state').textContent = delay ? t('jobs.polling') : '';
  if (delay) state.pollTimer = setTimeout(pollJobs, delay);
}

// A build takes minutes and only job statuses change meanwhile, so poll the small job endpoint and fetch the
// full details when a status changed, or every FULL_REFRESH_MS to pick up other visitors' uploads.
async function pollJobs() {
  state.pollTimer = null;
  const d = state.detail;
  if (Date.now() - state.detailAt >= FULL_REFRESH_MS) return refresh();
  const seq = state.refreshSeq;
  let polled;
  try {
    polled = await Promise.all(d.jobs.filter(isActiveJob).map((j) => getJob(j.id)));
  } catch {
    polled = null; // refresh() reports the error and keeps retrying
  }
  if (seq !== state.refreshSeq) return; // a refresh started meanwhile and schedules the next poll itself
  if (!polled || jobsChanged(d.jobs, polled)) refresh();
  else schedulePoll(d);
}

async function refresh() {
  const seq = ++state.refreshSeq;
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
  try {
    // the public poses with every full refresh (not on job polls); a server without them just shows none
    const [d, poses] = await Promise.all([getLandmark(landmarkId), getPoses(landmarkId).catch(() => undefined)]);
    if (seq !== state.refreshSeq) return;
    if (poses !== undefined) state.poses = parsePoses(poses);
    state.detail = d;
    state.detailAt = Date.now();
    $('landmark-content').hidden = false;
    showPageError('');
    render(d);
    schedulePoll(d);
  } catch (err) {
    if (seq !== state.refreshSeq) return;
    if (!state.detail) {
      showFatal(err instanceof ApiError && err.status === 404 ? t('landmark.notFound') : t('landmark.loadFailed', { message: err.message }));
      return;
    }
    showPageError(t('landmark.refreshFailed', { message: err.message }));
    if (state.detail.jobs.some(isActiveJob)) state.pollTimer = setTimeout(refresh, POLL_RUNNING_MS);
  }
}

function init() {
  if (!landmarkId) {
    showFatal(t('landmark.noId'));
    return;
  }
  $('user-id').value = readUserId();
  // Unticked on every load, whatever the browser restored: agreeing is always an explicit choice (D32).
  $('agree-terms').checked = false;
  state.shared = readShared();
  renderShared();
  $('upload-form').addEventListener('submit', onUpload);
  $('show-poses').addEventListener('change', () => state.viewer?.setCamerasVisible($('show-poses').checked));
  $('rebuild-btn').addEventListener('click', onRebuild);
  // Polling stops when no job is running, so catch up on other visitors' uploads when the tab comes back.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && state.detail && !state.pollTimer) refresh();
  });
  refresh();
}

init();
