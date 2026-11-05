// Landmark page: coverage compass, photo upload, contributors, jobs, models and the 3D viewer. Per photo it shows only
// the public poses (D30, GET /api/landmarks/{id}/poses): the photos uploaded under the user agreement and not
// withdrawn (D32), rounded by the server to 1 m and 1 degree (fields of view to 5 degrees), without uploader or time,
// as a table and as markers in the 3D viewer. Uploading needs ticking the agreement box (web/terms.html), which starts
// unticked on every page load and stays ticked for further uploads in the same page view; every upload sends the
// agreement's version (api.js TERMS_VERSION). The manage tokens (and public ids) of the uploaded photos stay in this
// browser (localStorage) so the uploader can find and withdraw them.

import {
  ApiError,
  JOB_KIND_TEXT,
  JOB_STATUS_TEXT,
  MODEL_KIND_TEXT,
  codeText,
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
import { compassSvg, sectorName } from './compass.js';
import { sectorSideBearing } from './framing.js';
import { GPS_RESIDUAL_LABEL, modelToShow, modelsKey, positionSigmaM, sortModels, stylizedView } from './models.js';
import {
  SNAPPED_CLOUD_NOTE,
  SOURCE_TEXT,
  cloudMarkers,
  countsText,
  coverageText,
  headingText,
  markersKey,
  parsePoses,
  positionText,
  rangeText,
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
import { applyDom } from './i18n.js';
import { el, fill, fmtCount, fmtMeters, fmtPercent, fmtTime, mountPrefs, progressBar } from './ui.js';

applyDom();
mountPrefs(document.getElementById('prefs'));

const MAX_FILES = 50;
const USER_KEY = 'crowd3d.user_id';
const SHARED_KEY_PREFIX = 'crowd3d.shared_location.'; // + landmark id: [{photo_id, token, filename, public_id}]
const VIEWER_UNAVAILABLE = '3D 查看器无法启动（浏览器可能不支持 WebGL，或网络无法加载 three.js）。';

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

function sectorLabel(d, index) {
  const s = d.sectors[index];
  return s ? sectorName(s.label) : `方向 ${index}`;
}

function buildTitle(d, kind, sector) {
  return kind === 'full' ? MODEL_KIND_TEXT.full : `${sectorLabel(d, sector)}立面`;
}

// A stylize job works on a published model: the full model (no sector) or a facade's.
function jobTitle(d, j) {
  if (j.kind !== 'stylize') return buildTitle(d, j.kind, j.sector);
  const model = j.sector === null || j.sector === undefined ? 'full' : 'face';
  return `${buildTitle(d, model, j.sector)}（${JOB_KIND_TEXT.stylize}）`;
}

// ---------- hero ----------

function renderHero(d) {
  document.title = `${d.name} · Crowd3D 众拍建模`;
  $('lm-name').textContent = d.name;
  $('lm-meta').textContent =
    `${d.lat.toFixed(5)}, ${d.lon.toFixed(5)} · 打卡半径 ${Math.round(d.radius_m)} 米 · ` +
    `${fmtCount(d.num_photos)} 张照片 · ${fmtCount(d.num_contributors)} 位参与者`;
  fill($('lm-progress'), progressBar(d.progress, '覆盖进度'));
  $('lm-progress-text').textContent = `覆盖进度 ${fmtPercent(d.progress)}`;
  $('lm-next').textContent = nextStep(d);
}

// ---------- coverage ----------

function renderCoverage(d) {
  const t = d.thresholds;
  const failed = failedFaceSectors(d.jobs);
  $('compass').innerHTML = compassSvg(d.sectors, { size: 260, faceMin: t.face_min_photos, failed });
  $('legend-failed').hidden = !failed.size;
  $('coverage-rule').textContent =
    `照片按拍摄者站的方向分成 ${t.num_sectors} 份。每个方向满 ${t.face_min_photos} 张生成该方向的立面模型；` +
    `${t.full_min_sectors} 个方向达标且共 ${t.full_min_photos} 张后生成完整模型。`;
}

// ---------- contributors ----------

// Nicknames only, by name: no per-person photo counts or ranks (D30), which would tie public photo poses to people.
function renderContributors(d) {
  const list = [...d.contributors].sort((a, b) => a.user_id.localeCompare(b.user_id));
  if (!list.length) {
    fill($('contributors'), el('li', { class: 'empty', text: '还没有人上传，成为第一个吧。' }));
    return;
  }
  const me = readUserId();
  fill(
    $('contributors'),
    list.map((c) =>
      el(
        'li',
        { class: c.user_id === me ? 'contributor is-me' : 'contributor' },
        c.user_id === me ? `${c.user_id}（你）` : c.user_id,
      ),
    ),
  );
}

// ---------- public photo poses ----------

function poseRows() {
  return state.poses?.photos ?? [];
}

// The table of public poses: source, position from the centre, heading, 2-sigma range; "你的" for this browser's own
// shared photos (their public ids came back with the upload).
function renderPoses() {
  const doc = state.poses;
  const rows = poseRows();
  $('poses-card').hidden = !doc;
  renderMarkerControls();
  if (!doc) return;
  $('poses-as-of').textContent = doc.as_of ? `${fmtTime(doc.as_of)}更新` : '';
  $('poses-count').textContent = doc.as_of
    ? `${countsText(doc.counts)}。`
    : '这里还没有建模完成过；公开的照片会在第一次建模结束后显示。';
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
          mine.has(r.id) ? el('span', { class: 'chip chip-mine', text: '你的' }) : null,
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
  note.textContent = shown && snapped ? SNAPPED_CLOUD_NOTE : '';
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
  if (j.status === 'queued') return `${fmtTime(j.created_at)}加入队列`;
  if (j.status === 'running') return `${fmtTime(j.started_at)}开始`;
  if (j.status === 'done') {
    const secs = j.started_at && j.finished_at ? Math.max(1, Math.round(j.finished_at - j.started_at)) : null;
    return `${fmtTime(j.finished_at)}完成${secs ? `，用时 ${secs} 秒` : ''}`;
  }
  return `${fmtTime(j.finished_at || j.created_at)}失败`;
}

function renderJobs(d) {
  if (!d.jobs.length) {
    fill($('job-list'), el('li', { class: 'empty', text: '暂无任务。某个方向的照片够了会自动开始。' }));
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
        el('div', { class: 'job-meta muted', text: `${j.num_photos} 张照片 · ${jobDetail(j)}` }),
        j.status === 'failed' ? jobFailure(d, j) : null,
      ),
    ),
  );
}

// Why it failed in Chinese, how it gets retried (newest attempt only), and the raw message for whoever debugs it.
function jobFailure(d, j) {
  return [
    el('div', { class: 'job-error', text: jobErrorText(j.error) }),
    isLatestFailure(d.jobs, j) ? el('div', { class: 'job-retry muted', text: `${retryText(d, j)}。` }) : null,
    j.error
      ? el('details', { class: 'job-error-detail muted' }, el('summary', { text: '技术信息' }), el('code', { text: j.error }))
      : null,
  ];
}

// ---------- models + viewer ----------

function modelMeta(m) {
  const parts = [`${fmtCount(m.num_points)} 个点`, `${m.num_registered}/${m.num_photos} 张照片定位成功`];
  if (m.georegistered) {
    let geo = '已对齐地理坐标';
    if (m.alignment_rmse_m !== null && m.alignment_rmse_m !== undefined) {
      geo += `（${GPS_RESIDUAL_LABEL} ${fmtMeters(m.alignment_rmse_m)}）`;
    }
    parts.push(geo);
    const sigma = positionSigmaM(m);
    if (sigma !== null) parts.push(`位置 ±${fmtMeters(sigma)}（1σ）`);
  } else {
    parts.push('未对齐地理坐标');
  }
  if (m.heading_mae_deg !== null && m.heading_mae_deg !== undefined) parts.push(`朝向误差 ${m.heading_mae_deg.toFixed(1)}°`);
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

  setViewerMessage('正在加载模型…');
  const viewer = await ensureViewer();
  if (state.shownModel !== m) return;
  if (!viewer) {
    setViewerMessage(VIEWER_UNAVAILABLE);
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
    setViewerMessage(`模型加载失败：${err.message || err}`);
  }
}

// ---------- manual rebuild ----------

function renderRebuild(d) {
  const targets = [];
  if (d.full_ready) targets.push(['full', MODEL_KIND_TEXT.full]);
  for (const s of d.sectors) {
    if (s.state === 'face_ready') targets.push([`face:${s.index}`, `${sectorName(s.label)}立面`]);
  }
  const key = targets.map((t) => t[0]).join();
  if (key === state.rebuildKey && $('rebuild-target').options.length) return; // keep an open dropdown open while polling
  state.rebuildKey = key;

  const select = $('rebuild-target');
  const previous = select.value;
  const options = targets.map(([value, text]) => el('option', { value, text }));
  if (!options.length) options.push(el('option', { value: '', text: '照片不足，暂不能重建' }));
  fill(select, options);
  if (targets.some((t) => t[0] === previous)) select.value = previous;
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
  status.textContent = '正在提交…';
  try {
    const job = await rebuild(landmarkId, kind, sector === undefined ? null : Number(sector));
    status.className = 'form-status is-ok';
    status.textContent = `已加入队列：${buildTitle(state.detail, job.kind, job.sector)}`;
    await refresh();
  } catch (err) {
    status.className = 'form-status is-error';
    status.textContent =
      err instanceof ApiError && err.status === 409 ? '照片数量还不够，暂时不能重建。' : `重建失败：${err.message}`;
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
  let summary = `${ok} 张通过${bad ? `，${bad} 张未通过` : ''}。`;
  if (res.triggered_jobs.length) {
    summary += ` 已开始生成：${res.triggered_jobs.map((j) => buildTitle(d, j.kind, j.sector)).join('、')}。`;
  }
  setUploadStatus(summary, bad && !ok ? 'is-error' : 'is-ok');

  // Two columns so the reasons get room in the narrow side column: photo (+ where it was taken), result (+ why).
  const rows = res.results.map((r) => {
    let where = null;
    if (r.sector !== null && r.sector !== undefined && d) {
      where = `${sectorLabel(d, r.sector)}${r.distance_m !== null && r.distance_m !== undefined ? ` · ${fmtMeters(r.distance_m)}` : ''}`;
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
        el('span', { class: `chip ${r.accepted ? 'chip-done' : 'chip-failed'}`, text: r.accepted ? '通过' : '未通过' }),
        r.accepted ? el('div', { class: 'muted share-state', text: r.share_location ? '已公开（可撤回）' : '未公开' }) : null,
        r.reasons.length ? el('div', { text: uploadReasonText(r) }) : null,
        r.warnings.length ? el('div', { class: 'muted', text: `提示：${r.warnings.map(codeText).join('、')}` }) : null,
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
        { class: 'table', 'aria-label': '上传结果' },
        el(
          'thead',
          {},
          el('tr', {}, ['照片', '结果'].map((h) => el('th', { scope: 'col', text: h }))),
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
    el('p', { class: 'shared-own-text', text: `你在这台设备上公开了 ${list.length} 张照片。撤回后，别人就看不到这些照片和它们的拍摄位置了。` }),
    el(
      'ul',
      { class: 'shared-list', 'aria-label': '公开的照片' },
      list.map((s) =>
        el(
          'li',
          { class: 'shared-item' },
          el('span', { class: 'shared-name', text: s.filename || '照片', title: s.filename || '' }),
          el('button', { type: 'button', class: 'btn btn-ghost btn-small', text: '撤回', onclick: () => withdraw([s]) }),
        ),
      ),
    ),
    list.length > 1
      ? el('button', { type: 'button', class: 'btn btn-ghost', text: '全部撤回', onclick: () => withdraw([...list]) })
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
  if (ok) parts.push(`已撤回 ${ok} 张照片`);
  if (lost) parts.push(`${lost} 张服务器已找不到，无法撤回`);
  if (refused) parts.push(`${refused} 张的撤回凭证已失效，照片和位置可能仍然公开`);
  if (failed) parts.push(`${failed} 张撤回失败，请稍后再试`);
  renderShared(`${parts.join('；')}。`);
  if (ok && state.detail) refresh(); // a withdrawn photo leaves the public poses at once
}

async function onUpload(event) {
  event.preventDefault();
  const userInput = $('user-id');
  const filesInput = $('files');
  const userId = userInput.value.trim();
  if (!userId || /[\u0000-\u001f\u007f]/.test(userId)) {
    setUploadStatus('请填写昵称（不超过 64 个字符）', 'is-error');
    userInput.focus();
    return;
  }
  const files = [...filesInput.files];
  if (!files.length) {
    setUploadStatus('请选择要上传的照片', 'is-error');
    filesInput.focus();
    return;
  }
  if (files.length > MAX_FILES) {
    setUploadStatus(`一次最多上传 ${MAX_FILES} 张，你选了 ${files.length} 张`, 'is-error');
    return;
  }
  const agree = $('agree-terms');
  if (agree.checked !== true) {
    // D32: no upload without the user agreement (the server refuses one without it too)
    setUploadStatus('请先阅读并同意《用户协议》', 'is-error');
    agree.focus();
    return;
  }
  saveUserId(userId);

  const form = $('upload-form');
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  button.textContent = '上传中…';
  form.setAttribute('aria-busy', 'true');
  setUploadStatus(`正在上传并检查 ${files.length} 张照片…`);
  try {
    const res = await uploadPhotos({ files, userId, landmarkId });
    renderUploadResults(res);
    filesInput.value = '';
    await refresh();
  } catch (err) {
    setUploadStatus(`上传失败：${err.message}`, 'is-error');
  } finally {
    button.disabled = false;
    button.textContent = '上传';
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
  $('lm-name').textContent = '无法打开地标';
  $('landmark-content').hidden = true;
  fill($('page-error'), el('span', { text }), ' ', el('a', { href: './', text: '返回地标列表' }));
  $('page-error').hidden = false;
}

function schedulePoll(d) {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
  const delay = pollDelay(d.jobs);
  $('poll-state').textContent = delay ? '自动刷新中' : '';
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
      showFatal(err instanceof ApiError && err.status === 404 ? '找不到这个地标。' : `加载失败：${err.message}`);
      return;
    }
    showPageError(`刷新失败：${err.message}`);
    if (state.detail.jobs.some(isActiveJob)) state.pollTimer = setTimeout(refresh, POLL_RUNNING_MS);
  }
}

function init() {
  if (!landmarkId) {
    showFatal('链接里缺少地标 id。');
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
