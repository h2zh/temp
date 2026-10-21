// Home page: map of landmarks, landmark cards with mini compasses, "新建地标" form.

import { ApiError, createLandmark, listLandmarks } from './api.js';
import { compassSvg } from './compass.js';
import { el, fill, fmtCount, fmtPercent, progressBar } from './ui.js';

const listEl = document.getElementById('landmark-list');
const countEl = document.getElementById('list-count');
const form = document.getElementById('new-landmark');
const statusEl = document.getElementById('new-status');

const DEFAULT_VIEW = { center: [31.2397, 121.4998], zoom: 13 };

// Leaflet is loaded from the page script, not from <head>: a render-blocking stylesheet or a deferred script
// on a slow or blocked CDN would hold back the whole page, including the landmark list that needs no map.
const LEAFLET = {
  css: { href: 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css', integrity: 'sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=' },
  js: { src: 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js', integrity: 'sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=' },
};
const MAP_SLOW_MS = 8000;

let map = null;
let markerLayer = null;
let pickMarker = null;
let landmarksShown = null; // the last list, so markers can be drawn once Leaflet arrives

function landmarkUrl(id) {
  return `landmark.html?id=${encodeURIComponent(id)}`;
}

function statusText(lm) {
  if (lm.latest_full_model) return { text: '完整模型已生成', cls: 'badge-ok' };
  if (lm.full_ready) return { text: '可生成完整模型', cls: 'badge-ok' };
  const faces = lm.sectors.filter((s) => s.has_face_model).length;
  if (faces) return { text: `${faces} 个立面已建模`, cls: 'badge-partial' };
  if (lm.num_photos) return { text: '收集照片中', cls: 'badge-partial' };
  return { text: '等待第一张照片', cls: 'badge-muted' };
}

function markerColor(lm) {
  if (lm.latest_full_model || lm.full_ready) return '#2f9e5b';
  return lm.num_photos ? '#e0a020' : '#8a94a6';
}

function loadAsset(tag, attrs) {
  return new Promise((resolve, reject) => {
    const node = el(tag, { ...attrs, crossorigin: '' });
    node.onload = resolve;
    node.onerror = reject;
    // First in <head>, so style.css still comes later in the cascade and overrides Leaflet's defaults.
    document.head.prepend(node);
  });
}

function mapMessage(text) {
  fill(document.getElementById('map'), el('p', { class: 'map-fallback muted', text }));
}

async function loadMap() {
  mapMessage('地图加载中…');
  const slow = setTimeout(() => mapMessage('地图加载较慢，可以先使用地标列表。'), MAP_SLOW_MS);
  try {
    await Promise.all([loadAsset('link', { rel: 'stylesheet', ...LEAFLET.css }), loadAsset('script', LEAFLET.js)]);
  } catch {
    // window.L stays unset; initMap shows the fallback text.
  }
  clearTimeout(slow);
  initMap();
  if (landmarksShown) renderMarkers(landmarksShown);
}

function initMap() {
  const L = window.L;
  const mapEl = document.getElementById('map');
  if (!L) {
    mapMessage('地图组件加载失败，可直接使用下方列表。');
    return;
  }
  mapEl.replaceChildren();
  map = L.map(mapEl, { scrollWheelZoom: false }).setView(DEFAULT_VIEW.center, DEFAULT_VIEW.zoom);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(map);
  markerLayer = L.layerGroup().addTo(map);
  map.on('click', (e) => pickLocation(e.latlng.lat, e.latlng.lng));
}

function pickLocation(lat, lon) {
  form.elements.lat.value = lat.toFixed(6);
  form.elements.lon.value = lon.toFixed(6);
  if (map) {
    const L = window.L;
    if (pickMarker) pickMarker.setLatLng([lat, lon]);
    else pickMarker = L.circleMarker([lat, lon], { radius: 6, color: '#2f6fdb', weight: 2, fillOpacity: 0.4 }).addTo(map);
  }
}

function popupContent(lm) {
  return el(
    'div',
    { class: 'map-popup' },
    el('strong', { text: lm.name }),
    el('div', { class: 'muted', text: `覆盖进度 ${fmtPercent(lm.progress)} · ${fmtCount(lm.num_photos)} 张照片` }),
    el('a', { href: landmarkUrl(lm.id), text: '查看详情 →' }),
  );
}

function renderMarkers(landmarks) {
  if (!map) return;
  const L = window.L;
  markerLayer.clearLayers();
  for (const lm of landmarks) {
    const color = markerColor(lm);
    L.circle([lm.lat, lm.lon], { radius: lm.radius_m, color, weight: 1, opacity: 0.6, fillOpacity: 0.08, interactive: false }).addTo(markerLayer);
    // bubblingMouseEvents: false so opening a popup doesn't also fill the form via the map click handler.
    // Leaflet sets string content as innerHTML, so the user-supplied name goes in as a text node.
    L.circleMarker([lm.lat, lm.lon], { radius: 9, color: '#ffffff', weight: 2, fillColor: color, fillOpacity: 1, bubblingMouseEvents: false })
      .bindPopup(() => popupContent(lm))
      .bindTooltip(() => el('span', { text: lm.name }))
      .addTo(markerLayer);
  }
  if (landmarks.length === 1) map.setView([landmarks[0].lat, landmarks[0].lon], 16);
  else if (landmarks.length > 1) map.fitBounds(L.latLngBounds(landmarks.map((lm) => [lm.lat, lm.lon])), { padding: [40, 40], maxZoom: 16 });
}

function card(lm) {
  const status = statusText(lm);
  return el(
    'article',
    { class: 'lm-card' },
    el('div', { class: 'lm-card-compass', html: compassSvg(lm.sectors, { size: 64, compact: true }) }),
    el(
      'div',
      { class: 'lm-card-body' },
      el('h3', { class: 'lm-card-title' }, el('a', { href: landmarkUrl(lm.id), text: lm.name, class: 'stretched-link' })),
      el(
        'div',
        { class: 'lm-card-progress' },
        progressBar(lm.progress, `${lm.name} 覆盖进度`),
        el('span', { class: 'lm-card-pct', text: fmtPercent(lm.progress) }),
      ),
      el(
        'div',
        { class: 'lm-card-meta' },
        el('span', { text: `${fmtCount(lm.num_photos)} 张照片` }),
        el('span', { text: `${fmtCount(lm.num_contributors)} 位参与者` }),
        el('span', { class: `badge ${status.cls}`, text: status.text }),
      ),
    ),
  );
}

async function loadLandmarks() {
  listEl.setAttribute('aria-busy', 'true');
  try {
    const landmarks = await listLandmarks();
    countEl.textContent = landmarks.length ? `共 ${landmarks.length} 个` : '';
    if (landmarks.length) fill(listEl, landmarks.map(card));
    else fill(listEl, el('p', { class: 'empty', text: '还没有地标。在“新建地标”里添加第一个吧。' }));
    landmarksShown = landmarks;
    renderMarkers(landmarks);
  } catch (err) {
    fill(
      listEl,
      el(
        'div',
        { class: 'notice notice-error', role: 'alert' },
        el('span', { text: `地标列表加载失败：${err.message}` }),
        el('button', { type: 'button', class: 'btn btn-ghost', text: '重试', onclick: loadLandmarks }),
      ),
    );
  } finally {
    listEl.setAttribute('aria-busy', 'false');
  }
}

function setStatus(text, kind = '', link = null) {
  statusEl.className = `form-status ${kind}`;
  fill(statusEl, text, link ? ' ' : null, link);
}

async function onCreate(event) {
  event.preventDefault(); // native validation (required/min/max) has already passed when submit fires
  const data = new FormData(form);
  const payload = {
    name: String(data.get('name')).trim(),
    lat: Number(data.get('lat')),
    lon: Number(data.get('lon')),
  };
  const radius = String(data.get('radius_m') || '').trim();
  if (radius) payload.radius_m = Number(radius);
  if (!payload.name) {
    setStatus('请填写名称', 'is-error');
    return;
  }

  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  setStatus('正在创建…');
  try {
    const lm = await createLandmark(payload);
    setStatus(`已创建「${lm.name}」。`, 'is-ok', el('a', { href: landmarkUrl(lm.id), text: '去上传照片 →' }));
    form.reset();
    if (pickMarker) {
      pickMarker.remove();
      pickMarker = null;
    }
    await loadLandmarks();
  } catch (err) {
    const text =
      err instanceof ApiError && err.status === 409
        ? '已存在相同标识的地标，请换一个名称。'
        : err instanceof ApiError && err.status === 422
          ? `填写内容有误：${err.message}`
          : `创建失败：${err.message}`;
    setStatus(text, 'is-error');
  } finally {
    button.disabled = false;
  }
}

function useMyLocation() {
  if (!navigator.geolocation) {
    setStatus('当前浏览器不支持定位', 'is-error');
    return;
  }
  setStatus('正在定位…');
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      pickLocation(pos.coords.latitude, pos.coords.longitude);
      if (map) map.setView([pos.coords.latitude, pos.coords.longitude], 16);
      setStatus('已填入当前位置');
    },
    () => setStatus('无法获取当前位置，请在地图上点击选择', 'is-error'),
    { enableHighAccuracy: true, timeout: 10000 },
  );
}

form.addEventListener('submit', onCreate);
document.getElementById('use-location').addEventListener('click', useMyLocation);
loadLandmarks();
loadMap();
