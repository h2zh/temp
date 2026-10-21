// Landmark page: coverage compass, photo upload, contributors, jobs, models and the 3D viewer.

import {
  ApiError,
  JOB_STATUS_TEXT,
  MODEL_KIND_TEXT,
  codeText,
  getJob,
  getLandmark,
  getModelCameras,
  isActiveJob,
  jobErrorText,
  rebuild,
  uploadPhotos,
  uploadReasonText,
} from './api.js';
import { compassSvg, sectorName } from './compass.js';
import { photoSideBearing } from './framing.js';
import { modelToShow, sortModels } from './models.js';
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
import { el, fill, fmtCount, fmtDeg, fmtMeters, fmtPercent, fmtTime, progressBar } from './ui.js';

const MAX_FILES = 50;
const USER_KEY = 'crowd3d.user_id';
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
  shownCamerasFromPhotos: false, // the model's own cameras were unavailable; markers come from d.photos
  modelCameras: new Map(), // model id -> its cameras (models never change)
  userPickedModel: false,
  modelsKey: '',
  rebuildKey: '',
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

function sectorLabel(d, index) {
  const s = d.sectors[index];
  return s ? sectorName(s.label) : `方向 ${index}`;
}

function buildTitle(d, kind, sector) {
  return kind === 'full' ? MODEL_KIND_TEXT.full : `${sectorLabel(d, sector)}立面`;
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

function renderContributors(d) {
  const list = [...d.contributors].sort((a, b) => b.photos - a.photos || a.user_id.localeCompare(b.user_id));
  if (!list.length) {
    fill($('contributors'), el('li', { class: 'empty', text: '还没有人上传，成为第一个吧。' }));
    return;
  }
  const me = readUserId();
  const max = list[0].photos || 1;
  fill(
    $('contributors'),
    list.map((c, i) =>
      el(
        'li',
        { class: c.user_id === me ? 'leader is-me' : 'leader' },
        el('span', { class: 'leader-rank', text: String(i + 1) }),
        el('span', { class: 'leader-name', text: c.user_id === me ? `${c.user_id}（你）` : c.user_id }),
        el('span', { class: 'leader-bar', 'aria-hidden': 'true' }, el('span', { style: `width: ${Math.round((c.photos / max) * 100)}%` })),
        el('span', { class: 'leader-count', text: `${fmtCount(c.photos)} 张` }),
      ),
    ),
  );
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
          el('span', { class: 'job-title', text: buildTitle(d, j.kind, j.sector) }),
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
    if (m.alignment_rmse_m !== null && m.alignment_rmse_m !== undefined) geo += `（误差 ${fmtMeters(m.alignment_rmse_m)}）`;
    parts.push(geo);
  } else {
    parts.push('未对齐地理坐标');
  }
  if (m.heading_mae_deg !== null && m.heading_mae_deg !== undefined) parts.push(`朝向误差 ${m.heading_mae_deg.toFixed(1)}°`);
  return parts.join(' · ');
}

function renderModels(d) {
  const models = sortModels(d.models);
  // Polling re-renders every 3 s; only rebuild the buttons when the models changed so keyboard focus survives.
  const key = models.map((m) => m.id).join();
  if (key !== state.modelsKey) {
    state.modelsKey = key;
    fillModelList(d, models);
  }
  if (!models.length) return;

  const next = modelToShow(models, state.shownModel, state.userPickedModel);
  if (next) showModel(next);
  else if (state.shownCamerasFromPhotos && state.viewer) state.viewer.setCameras(photoCameras(d, state.shownModel));
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
      ),
    ),
  );
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

  setViewerMessage('正在加载模型…');
  const [viewer, own] = await Promise.all([ensureViewer(), cachedModelCameras(m)]);
  if (state.shownModel !== m) return;
  if (!viewer) {
    setViewerMessage(VIEWER_UNAVAILABLE);
    return;
  }
  state.shownCamerasFromPhotos = !own;
  const cameras = own ?? photoCameras(state.detail, m);
  // Seen from where the photos were taken: a facade from its sector, a full model from the side its photos
  // cluster on, or from the viewer's default when they surround it.
  const sector = m.kind === 'face' ? state.detail.sectors[m.sector] : null;
  const viewFromDeg = sector ? sector.bearing_center_deg : photoSideBearing(cameras.map((c) => c.enu));
  try {
    viewer.setCamerasVisible($('show-cameras').checked);
    const result = await viewer.load(m.url, { ...(viewFromDeg === null ? {} : { viewFromDeg }), cameras });
    if (!result) return; // superseded by a newer click
    setViewerMessage('');
  } catch (err) {
    setViewerMessage(`模型加载失败：${err.message || err}`);
  }
}

// The markers show where the model on screen placed its photos. Each model is registered to the map on its
// own, so poses from another model of the same photo can sit metres away from this point cloud.
async function cachedModelCameras(m) {
  if (!state.modelCameras.has(m.id)) {
    try {
      const cameras = await getModelCameras(m);
      if (cameras) state.modelCameras.set(m.id, cameras);
      else return null;
    } catch (err) {
      console.warn('model cameras unavailable:', err);
      return null;
    }
  }
  return state.modelCameras.get(m.id);
}

// Fallback when the model's own cameras cannot be read: PhotoInfo estimates (newest model per photo).
function photoCameras(d, m) {
  return d.photos
    .filter((p) => p.estimated_enu && (m.kind === 'full' || p.sector === m.sector))
    .map((p) => ({ enu: p.estimated_enu, heading_deg: p.estimated_heading_deg }));
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

// ---------- photos table ----------

function renderPhotos(d) {
  // The API lists only the newest photos of a busy landmark.
  $('photos-title').textContent =
    d.photos.length < d.num_photos
      ? `照片方位（最新 ${fmtCount(d.photos.length)} 张 / 共 ${fmtCount(d.num_photos)} 张）`
      : `照片方位（${fmtCount(d.photos.length)}）`;
  if (!d.photos.length) {
    fill($('photo-rows'), el('tr', {}, el('td', { colspan: 7, class: 'empty', text: '还没有照片。' })));
    return;
  }
  fill(
    $('photo-rows'),
    d.photos.map((p) =>
      el(
        'tr',
        {},
        el('td', { text: p.user_id, title: p.user_id }),
        el('td', { text: p.sector === null || p.sector === undefined ? '—' : sectorLabel(d, p.sector) }),
        el('td', { text: fmtMeters(p.distance_m) }),
        el('td', { text: fmtDeg(p.heading_deg) }),
        el('td', { text: fmtDeg(p.estimated_heading_deg) }),
        el('td', { text: p.estimated_enu ? `${p.estimated_enu[0].toFixed(1)}, ${p.estimated_enu[1].toFixed(1)}` : '—' }),
        el('td', { text: fmtTime(p.uploaded_at) }),
      ),
    ),
  );
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
  renderJobs(d);
  renderModels(d);
  renderRebuild(d);
  renderPhotos(d);
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
    const d = await getLandmark(landmarkId);
    if (seq !== state.refreshSeq) return;
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
  $('upload-form').addEventListener('submit', onUpload);
  $('rebuild-btn').addEventListener('click', onRebuild);
  $('show-cameras').addEventListener('change', (e) => state.viewer?.setCamerasVisible(e.target.checked));
  // Polling stops when no job is running, so catch up on other visitors' uploads when the tab comes back.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && state.detail && !state.pollTimer) refresh();
  });
  refresh();
}

init();
