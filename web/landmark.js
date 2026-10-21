// Landmark page: coverage compass, photo upload, contributors, jobs, models and the 3D viewer.

import {
  ApiError,
  JOB_STATUS_TEXT,
  MODEL_KIND_TEXT,
  codeText,
  getLandmark,
  isActiveJob,
  rebuild,
  uploadPhotos,
} from './api.js';
import { compassSvg, sectorName } from './compass.js';
import { el, fill, fmtCount, fmtDeg, fmtMeters, fmtPercent, fmtTime, progressBar } from './ui.js';

const POLL_MS = 3000;
const MAX_FILES = 50;
const USER_KEY = 'crowd3d.user_id';

const $ = (id) => document.getElementById(id);
const landmarkId = new URLSearchParams(location.search).get('id');

const state = {
  detail: null,
  pollTimer: null,
  refreshSeq: 0,
  viewer: null,
  viewerPromise: null,
  shownModel: null,
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

function nextStep(d) {
  const t = d.thresholds;
  if (d.jobs.some((j) => j.kind === 'full' && isActiveJob(j))) return '完整模型正在生成，请稍候…';
  if (d.latest_full_model) return `完整模型已生成（${d.latest_full_model.num_photos} 张照片参与）。继续上传可以让模型更完整。`;
  if (d.full_ready) return '照片已足够生成完整模型，等待生成中。';
  const ready = d.sectors.filter((s) => s.state === 'face_ready').length;
  const needSectors = Math.max(0, t.full_min_sectors - ready);
  const needPhotos = Math.max(0, t.full_min_photos - d.num_photos);
  const parts = [];
  if (needSectors) {
    const suggest = d.sectors
      .filter((s) => s.state !== 'face_ready')
      .sort((a, b) => b.count - a.count || a.index - b.index)
      .slice(0, needSectors)
      .map((s) => sectorName(s.label));
    parts.push(`还需 ${needSectors} 个方向各达到 ${t.face_min_photos} 张（建议补拍：${suggest.join('、')}）`);
  }
  if (needPhotos) parts.push(`总照片还差 ${needPhotos} 张`);
  return parts.length ? `距离完整模型：${parts.join('，')}。` : '';
}

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
  $('compass').innerHTML = compassSvg(d.sectors, { size: 260, faceMin: t.face_min_photos });
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
        j.status === 'failed' && j.error ? el('div', { class: 'job-error', text: j.error }) : null,
      ),
    ),
  );
}

// ---------- models + viewer ----------

function sortModels(models) {
  return [...models].sort(
    (a, b) =>
      (a.kind === 'full' ? 0 : 1) - (b.kind === 'full' ? 0 : 1) ||
      (a.sector ?? -1) - (b.sector ?? -1) ||
      b.created_at - a.created_at,
  );
}

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

  // First visit: show the best model. Later: follow rebuilds of the model on screen.
  const shown = state.shownModel;
  if (!shown) {
    showModel(models[0]);
  } else {
    const same = models.find((m) => m.kind === shown.kind && m.sector === shown.sector);
    if (same && same.id !== shown.id) showModel(same);
    else updateCameras();
  }
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
            onclick: () => showModel(m),
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
        setViewerMessage('3D 查看器无法启动（浏览器可能不支持 WebGL，或网络无法加载 three.js）。');
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
  const viewer = await ensureViewer();
  if (!viewer || state.shownModel !== m) return;
  try {
    const result = await viewer.load(m.url);
    if (!result) return; // superseded by a newer click
    setViewerMessage('');
    updateCameras();
  } catch (err) {
    setViewerMessage(`模型加载失败：${err.message || err}`);
  }
}

function updateCameras() {
  const m = state.shownModel;
  if (!state.viewer || !m || !state.detail) return;
  const photos = state.detail.photos.filter(
    (p) => p.estimated_enu && (m.kind === 'full' || p.sector === m.sector),
  );
  state.viewer.setCameras(photos.map((p) => ({ enu: p.estimated_enu, heading_deg: p.estimated_heading_deg })));
  state.viewer.setCamerasVisible($('show-cameras').checked);
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
  $('photos-title').textContent = `照片方位（${d.photos.length}）`;
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
        el('td', { text: p.user_id }),
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

  const rows = res.results.map((r) => {
    const notes = [];
    if (r.reasons.length) notes.push(el('span', { text: r.reasons.map(codeText).join('、') }));
    if (r.warnings.length) notes.push(el('span', { class: 'muted', text: `提示：${r.warnings.map(codeText).join('、')}` }));
    let where = '—';
    if (r.sector !== null && r.sector !== undefined && d) {
      where = `${sectorLabel(d, r.sector)}${r.distance_m !== null && r.distance_m !== undefined ? ` · ${fmtMeters(r.distance_m)}` : ''}`;
    }
    return el(
      'tr',
      {},
      el('td', { class: 'cell-file', text: r.filename }),
      el('td', {}, el('span', { class: `chip ${r.accepted ? 'chip-done' : 'chip-failed'}`, text: r.accepted ? '通过' : '未通过' })),
      el('td', { class: 'cell-nowrap', text: where }),
      el('td', { class: 'cell-notes' }, notes.length ? notes : '—'),
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
          el('tr', {}, ['文件', '结果', '方向', '说明'].map((h) => el('th', { scope: 'col', text: h }))),
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
  const active = d.jobs.some(isActiveJob);
  $('poll-state').textContent = active ? '自动刷新中' : '';
  if (active) state.pollTimer = setTimeout(refresh, POLL_MS);
}

async function refresh() {
  const seq = ++state.refreshSeq;
  clearTimeout(state.pollTimer);
  try {
    const d = await getLandmark(landmarkId);
    if (seq !== state.refreshSeq) return;
    state.detail = d;
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
    if (state.detail.jobs.some(isActiveJob)) state.pollTimer = setTimeout(refresh, POLL_MS);
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
  refresh();
}

init();
