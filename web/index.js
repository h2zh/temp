// Home page: map of landmarks, landmark cards with mini compasses, the "new landmark" form. Texts in the page language
// (web/strings.js home.*).

import { ApiError, createLandmark, listLandmarks } from './api.js';
import { compassSvg } from './compass.js';
import { applyDom, t } from './i18n.js';
import { el, fill, fmtPercent, mountPrefs, progressBar } from './ui.js';

applyDom();
mountPrefs(document.getElementById('prefs'));

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

function photosText(n) {
  return t('common.photoCount', { n: n ?? 0 });
}

function statusText(lm) {
  if (lm.latest_full_model) return { text: t('home.status.full'), cls: 'badge-ok' };
  if (lm.full_ready) return { text: t('home.status.fullReady'), cls: 'badge-ok' };
  const faces = lm.sectors.filter((s) => s.has_face_model).length;
  if (faces) return { text: t('home.status.faces', { n: faces }), cls: 'badge-partial' };
  if (lm.num_photos) return { text: t('home.status.collecting'), cls: 'badge-partial' };
  return { text: t('home.status.waiting'), cls: 'badge-muted' };
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
  mapMessage(t('home.map.loading'));
  const slow = setTimeout(() => mapMessage(t('home.map.slow')), MAP_SLOW_MS);
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
    mapMessage(t('home.map.failed'));
    return;
  }
  mapEl.replaceChildren();
  // Leaflet's own zoom control titles its buttons in English: this one says it in the page language.
  map = L.map(mapEl, { scrollWheelZoom: false, zoomControl: false }).setView(DEFAULT_VIEW.center, DEFAULT_VIEW.zoom);
  L.control.zoom({ zoomInTitle: t('map.zoomIn'), zoomOutTitle: t('map.zoomOut') }).addTo(map);
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
    el('div', { class: 'muted', text: `${t('common.coverageValue', { percent: fmtPercent(lm.progress) })} · ${photosText(lm.num_photos)}` }),
    el('a', { href: landmarkUrl(lm.id), text: t('home.details') }),
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
        progressBar(lm.progress, t('home.card.progressLabel', { name: lm.name })),
        el('span', { class: 'lm-card-pct', text: fmtPercent(lm.progress) }),
      ),
      el(
        'div',
        { class: 'lm-card-meta' },
        el('span', { text: photosText(lm.num_photos) }),
        el('span', { text: t('common.contributorCount', { n: lm.num_contributors ?? 0 }) }),
        el('span', { class: `badge ${status.cls}`, text: status.text }),
      ),
    ),
  );
}

async function loadLandmarks() {
  listEl.setAttribute('aria-busy', 'true');
  try {
    const landmarks = await listLandmarks();
    countEl.textContent = landmarks.length ? t('home.list.count', { n: landmarks.length }) : '';
    if (landmarks.length) fill(listEl, landmarks.map(card));
    else fill(listEl, el('p', { class: 'empty', text: t('home.list.empty') }));
    landmarksShown = landmarks;
    renderMarkers(landmarks);
  } catch (err) {
    fill(
      listEl,
      el(
        'div',
        { class: 'notice notice-error', role: 'alert' },
        el('span', { text: t('home.list.loadFailed', { message: err.message }) }),
        el('button', { type: 'button', class: 'btn btn-ghost', text: t('common.retry'), onclick: loadLandmarks }),
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
    setStatus(t('home.create.nameMissing'), 'is-error');
    return;
  }

  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  setStatus(t('home.create.creating'));
  try {
    const lm = await createLandmark(payload);
    setStatus(t('home.create.done', { name: lm.name }), 'is-ok', el('a', { href: landmarkUrl(lm.id), text: t('home.create.upload') }));
    form.reset();
    if (pickMarker) {
      pickMarker.remove();
      pickMarker = null;
    }
    await loadLandmarks();
  } catch (err) {
    const text =
      err instanceof ApiError && err.status === 409
        ? t('home.create.duplicate')
        : err instanceof ApiError && err.status === 422
          ? t('home.create.invalid', { message: err.message })
          : t('home.create.failed', { message: err.message });
    setStatus(text, 'is-error');
  } finally {
    button.disabled = false;
  }
}

function useMyLocation() {
  if (!navigator.geolocation) {
    setStatus(t('home.locate.unsupported'), 'is-error');
    return;
  }
  setStatus(t('home.locate.locating'));
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      pickLocation(pos.coords.latitude, pos.coords.longitude);
      if (map) map.setView([pos.coords.latitude, pos.coords.longitude], 16);
      setStatus(t('home.locate.done'));
    },
    () => setStatus(t('home.locate.failed'), 'is-error'),
    { enableHighAccuracy: true, timeout: 10000 },
  );
}

form.addEventListener('submit', onCreate);
document.getElementById('use-location').addEventListener('click', useMyLocation);
loadLandmarks();
loadMap();
