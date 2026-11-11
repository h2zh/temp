// Run from the repo root: node --test 'web/tests/*.test.mjs'
// Imports the real page scripts against the fake DOM in fake-dom.mjs, with fetch and Leaflet mocked.
import assert from 'node:assert/strict';
import { afterEach, mock, test } from 'node:test';

import { missing, t } from '../i18n.js';
import { FakeElement, inLang, installDom, settle, text, useDom } from './fake-dom.mjs';

const realFetch = globalThis.fetch;
const realWarn = console.warn;
let doc = null;
let page = 0;
let fakedTimers = false; // fakeTimers enabled the fake timers, not the test itself

afterEach(() => {
  mock.timers.reset();
  fakedTimers = false;
  globalThis.fetch = realFetch;
  console.warn = realWarn;
  delete globalThis.L;
  delete globalThis.localStorage;
  delete globalThis.crowd3dPrefs;
  doc?.restore();
  doc = null;
});

/**
 * The landmark page's timers are fake in every test (called by whatever opens the page): its job polls, and the asks
 * it repeats after an upload without an answer (D34), run only when a test ticks the clock. A real one would fire
 * seconds later, into another test's page. A test that also reads the clock enables its own before opening the page.
 * fresh: the page that was open is closed, and its timers go with it, as a browser's do.
 */
function fakeTimers({ fresh = false } = {}) {
  if (fresh && fakedTimers) {
    mock.timers.reset();
    fakedTimers = false;
  }
  try {
    mock.timers.enable({ apis: ['setTimeout'] });
    fakedTimers = true;
  } catch {
    // enabled already: by the test, or for the page before
  }
}

/** A localStorage stand-in (node has none without a flag); .map is what it holds. */
function fakeStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    map,
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
  };
}

// D33: this device's registration (web/landmark.js), as the page keeps it in localStorage
const CONTRIBUTOR_KEY = 'crowd3d.contributor';
const TOKEN = 'f0'.repeat(32);

/** The localStorage of a device that registered nickname under the agreement version terms, plus other entries. */
function registeredStorage(nickname = 'u', extra = {}, terms = 'terms-v1') {
  return fakeStorage({ [CONTRIBUTOR_KEY]: JSON.stringify({ token: TOKEN, nickname, terms }), ...extra });
}

/** fetch mock: routes maps a path (or path prefix ending in '*') to a body or a function of (url, init). */
function mockFetch(routes) {
  const calls = [];
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url, method: init.method ?? 'GET' });
    const key = Object.keys(routes).find((k) => (k.endsWith('*') ? url.startsWith(k.slice(0, -1)) : url === k));
    if (!key) return new Response(JSON.stringify({ detail: 'Not Found' }), { status: 404 });
    const body = typeof routes[key] === 'function' ? routes[key](url, init) : routes[key];
    return new Response(JSON.stringify(body), { status: 200 });
  };
  return calls;
}

/** Every element in the document whose markup was parsed from a string. */
function parsedMarkup(root) {
  return [root, ...root.descendants()].filter((n) => n.innerHTMLSets > 0).map((n) => n.innerHTML);
}

function fakeLeaflet({ mapWidth = 324 } = {}) {
  const tooltips = [];
  const zooms = []; // the options of each zoom control
  const popups = []; // the options of each bound popup
  const handlers = {}; // the map's (and the layers') event handlers by type
  const layer = () => {
    const o = {
      addTo: () => o,
      setView: () => o,
      on: (type, fn) => {
        (handlers[type] ??= []).push(fn);
        return o;
      },
      getSize: () => ({ x: mapWidth, y: 300 }),
      bindPopup: (content, options) => {
        popups.push(options);
        return o;
      },
      bindTooltip: (content) => {
        tooltips.push(content);
        return o;
      },
      clearLayers() {},
      fitBounds: () => o,
    };
    return o;
  };
  return {
    tooltips,
    zooms,
    popups,
    handlers,
    L: {
      map: (el, options) => {
        zooms.push({ map: options });
        return layer();
      },
      tileLayer: layer,
      layerGroup: layer,
      circle: layer,
      circleMarker: layer,
      latLngBounds: () => ({}),
      control: {
        zoom: (options) => {
          zooms.push(options);
          return layer();
        },
      },
    },
  };
}

const NAME = '<img src=x onerror="window.__xss=1">塔';
const SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'].map((label, index) => ({
  index,
  label,
  count: index < 6 ? 5 : index === 6 ? 4 : 0,
  state: index < 7 ? 'face_ready' : 'locked',
  bearing_center_deg: index * 45,
  has_face_model: index < 2,
}));

function summary(over = {}) {
  return {
    id: 'lm',
    name: NAME,
    lat: 43.773,
    lon: 11.2547,
    radius_m: 150,
    num_photos: 34,
    num_contributors: 2,
    progress: 0.8,
    full_ready: true,
    sectors: SECTORS,
    latest_full_model: null,
    ...over,
  };
}

test('home page: the list renders before Leaflet loads, and map tooltips never parse names as HTML', async () => {
  doc = installDom();
  const styleCss = new FakeElement('link');
  styleCss.setAttribute('href', 'style.css');
  doc.head.append(styleCss);
  mockFetch({ '/api/landmarks': { landmarks: [summary()] } });
  await import(`../index.js?page=${++page}`);
  await settle();

  // Leaflet is requested from the page script (pinned, with SRI), and the list is there before it arrives.
  // Its stylesheet goes before style.css, which overrides some Leaflet defaults (dark map background).
  const css = doc.head.querySelector('link[rel="stylesheet"]');
  const js = doc.head.querySelector('script');
  assert.ok(doc.head.children.indexOf(css) < doc.head.children.indexOf(styleCss));
  assert.equal(css.getAttribute('href'), 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css');
  assert.match(css.getAttribute('integrity'), /^sha256-/);
  assert.equal(js.tagName, 'SCRIPT');
  assert.equal(js.getAttribute('src'), 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js');
  assert.match(js.getAttribute('integrity'), /^sha256-/);
  assert.match(text(doc.getElementById('landmark-list')), /<img src=x onerror="window.__xss=1">塔/);
  assert.equal(text(doc.getElementById('map')), '地图加载中…');

  const leaflet = fakeLeaflet();
  globalThis.L = leaflet.L;
  css.onload();
  js.onload();
  await settle();

  assert.equal(leaflet.tooltips.length, 1);
  const tip = leaflet.tooltips[0];
  // Leaflet assigns string content to innerHTML; anything else is appended as a node.
  const content = typeof tip === 'function' ? tip() : tip;
  assert.ok(content instanceof FakeElement, `tooltip content must be a DOM node, got ${typeof content}`);
  assert.equal(content.textContent, NAME);
  assert.equal(content.innerHTMLSets, 0);
  for (const markup of parsedMarkup(doc.body)) assert.ok(!markup.includes('<img'), markup.slice(0, 80));
});

test('home page: a CDN failure leaves the list working and says the map is unavailable', async () => {
  doc = installDom();
  mockFetch({ '/api/landmarks': { landmarks: [summary({ name: '塔' })] } });
  await import(`../index.js?page=${++page}`);
  await settle();
  doc.head.querySelector('script').onerror();
  await settle();
  assert.equal(text(doc.getElementById('map')), '地图组件加载失败，可直接使用下方列表。');
  assert.match(text(doc.getElementById('landmark-list')), /塔/);
});

const FAILED_FULL = {
  id: 'jf',
  landmark_id: 'lm',
  kind: 'full',
  sector: null,
  status: 'failed',
  created_at: 20,
  started_at: 21,
  finished_at: 22,
  error: 'structure-from-motion found no initial image pair; photos may not overlap enough',
  num_photos: 30,
};

function job(over) {
  return { landmark_id: 'lm', created_at: 10, started_at: 11, finished_at: 12, error: null, num_photos: 5, ...over };
}

function model(id, kind, sector) {
  return {
    id,
    kind,
    sector,
    url: `/files/models/lm/${id}.ply`,
    created_at: 12,
    num_photos: 5,
    num_registered: 5,
    num_points: 1000,
    georegistered: true,
    alignment_rmse_m: 3,
    heading_mae_deg: 4,
  };
}

function landmarkDetail(over = {}) {
  return {
    ...summary(),
    contributors: [{ user_id: 'u' }],
    models: [model('m0', 'face', 0), model('m1', 'face', 1)],
    jobs: [
      FAILED_FULL,
      job({ id: 'jw', kind: 'face', sector: 6, status: 'failed', error: 'only 2 of 6 photos could be registered; need at least 3 overlapping views' }),
      job({ id: 'j0', kind: 'face', sector: 0, status: 'done' }),
      job({ id: 'j1', kind: 'face', sector: 1, status: 'done' }),
    ],
    thresholds: { face_min_photos: 4, full_min_photos: 24, full_min_sectors: 6, num_sectors: 8 },
    ...over,
  };
}

async function openLandmark(routes) {
  fakeTimers();
  doc = installDom({ search: '?id=lm' });
  console.warn = () => {};
  const calls = mockFetch(routes);
  await import(`../landmark.js?page=${++page}`);
  await settle();
  return calls;
}

test('every page script translates its page and draws the language switch in #prefs (D32)', async () => {
  mock.timers.enable({ apis: ['setTimeout'] }); // the home page's slow-map notice never fires
  console.warn = () => {};
  for (const [script, search, routes] of [
    ['index.js', '', { '/api/landmarks': { landmarks: [summary()] } }],
    ['landmark.js', '?id=lm', { '/api/landmarks/lm': landmarkDetail() }],
    ['terms.js', '', {}],
  ]) {
    doc?.restore();
    doc = installDom({ search });
    mockFetch(routes);
    // an English page as boot.js leaves it: hidden until translated
    const html = new FakeElement('html');
    const classes = new Set(['i18n-pending']);
    html.classList = { add: (c) => classes.add(c), remove: (c) => classes.delete(c) };
    html.dataset.lang = 'en';
    doc.documentElement = html;
    const back = doc.createElement('a');
    back.setAttribute('data-i18n', 'common.backToAll');
    back.textContent = '← 全部地标';
    doc.body.append(back);
    const calls = [];
    globalThis.crowd3dPrefs = { langChoice: () => 'en', setLang: (c) => calls.push(c), themeChoice: () => 'system', setTheme: () => {} };
    await import(`../${script}?page=${++page}`);
    assert.equal(back.textContent, '← All landmarks', `${script} translates the page`);
    assert.equal(html.getAttribute('lang'), 'en', script);
    assert.ok(!classes.has('i18n-pending'), `${script} shows the translated page`);
    const select = doc.getElementById('prefs').querySelector('select');
    assert.ok(select, `${script} draws the switch in #prefs`);
    assert.equal(select.value, 'en', script);
    select.value = 'zh';
    select.dispatch('change');
    assert.deepEqual(calls, ['zh'], script);
    await settle();
  }
});

test('landmark page: failed builds are explained in Chinese with how they get retried', async () => {
  await openLandmark({ '/api/landmarks/lm': landmarkDetail() });

  const hero = text(doc.getElementById('lm-next'));
  assert.doesNotMatch(hero, /等待生成中/);
  assert.match(hero, /^完整模型生成失败：照片之间重叠太少/);
  assert.match(hero, /用现有的 34 张照片马上重试/);

  const compass = doc.getElementById('compass').innerHTML;
  assert.match(compass, /西（W）：4 张 \/ 4，照片已够，立面生成失败/);
  assert.equal(doc.getElementById('legend-failed').hidden, false);

  const rows = doc.getElementById('job-list').querySelectorAll('li');
  const west = text(rows[1]);
  assert.match(west, /6 张照片里只有 2 张能互相对上/);
  assert.match(west, /这个方向再多几张照片后会自动重试/);
  // The English message is still there for whoever debugs it, folded away.
  const raw = rows[1].querySelector('details code');
  assert.equal(raw.textContent, 'only 2 of 6 photos could be registered; need at least 3 overlapping views');
  assert.equal(rows[2].querySelector('.job-error'), null);
});

test('landmark page: no failure marks while nothing failed', async () => {
  await openLandmark({ '/api/landmarks/lm': landmarkDetail({ jobs: [job({ id: 'j0', kind: 'face', sector: 0, status: 'done' })] }) });
  assert.equal(doc.getElementById('legend-failed').hidden, true);
  assert.doesNotMatch(doc.getElementById('compass').innerHTML, /compass-fail/);
});

test('landmark page: each model links its stylized version with the label; stylize jobs are named', async () => {
  mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_000_000 }); // a running job: the page polls
  const refused = {
    status: 'ready',
    url: '/files/models/lm/m0.glb',
    registration: { accepted: false, reason: 'ambiguous' },
  };
  await openLandmark({
    '/api/landmarks/lm': landmarkDetail({
      models: [
        { ...model('m0', 'face', 0), stylized: refused },
        { ...model('m1', 'face', 1), stylized: { status: 'pending', url: null, registration: null } },
      ],
      jobs: [
        job({ id: 's1', kind: 'stylize', sector: 1, status: 'running', finished_at: null, model_id: 'm1' }),
        job({ id: 's0', kind: 'stylize', sector: 0, status: 'done', model_id: 'm0' }),
        job({ id: 'j0', kind: 'face', sector: 0, status: 'done' }),
      ],
    }),
    '/api/jobs/s1': job({ id: 's1', kind: 'stylize', sector: 1, status: 'running', finished_at: null, model_id: 'm1' }),
  });
  const items = doc.getElementById('model-list').querySelectorAll('li');
  const link = items[0].querySelector('a.model-stylized');
  assert.equal(link.getAttribute('href'), 'stylized.html?model=%2Ffiles%2Fmodels%2Flm%2Fm0.glb');
  assert.equal(link.textContent, '风格化模型 · 素模（照片未对齐）');
  assert.match(link.getAttribute('class'), /is-unaligned/);
  assert.equal(items[1].querySelector('a.model-stylized'), null);
  assert.equal(text(items[1].querySelector('.model-stylized')), '风格化生成中');
  const titles = doc.getElementById('job-list').querySelectorAll('.job-title').map((n) => n.textContent);
  assert.deepEqual(titles, ['东北立面（风格化）', '北立面（风格化）', '北立面']);
  assert.equal(doc.getElementById('legend-failed').hidden, true);
});

test('landmark page: the model says GPS alignment residual, and a position 1-sigma only once calibrated', async () => {
  const geo = (calibrated) => ({
    level: 'full',
    sigma: { yaw_deg: 7.9, log_scale: 0.36, east_m: 3.5, north_m: 3.5, up_m: 1.5, tilt_deg: 2.2 },
    source: { yaw: 'gps+compass', scale: 'gps', xy: 'gps', up: 'camera-height', gravity: 'camera-rights' },
    calibration_id: 'dev-core5-v1',
    position_sigma_m: { horizontal: 12.4, vertical: 1.9 },
    calibrated,
  });
  await openLandmark({
    '/api/landmarks/lm': landmarkDetail({
      models: [{ ...model('m0', 'face', 0), geo: geo(false) }, { ...model('m1', 'face', 1), geo: geo(true) }],
    }),
  });
  const metas = doc.getElementById('model-list').querySelectorAll('.model-meta').map((n) => n.textContent);
  assert.equal(metas[0], '1,000 个点 · 5/5 张照片定位成功 · 已对齐地理坐标（GPS 对齐残差 3.0 米） · 朝向误差 4.0°');
  assert.doesNotMatch(metas[0], /位置|（误差/);
  assert.match(metas[1], /已对齐地理坐标（GPS 对齐残差 3\.0 米） · 位置 ±12 米（1σ） · /);
});

test('landmark page: no per-photo data from the detail, even from a server that still sends it, and no camera requests (D30)', async () => {
  // An older server's detail listed each photo with its uploader, position and the SfM camera's exact pose, and a
  // model could name a cameras_url; the page reads neither. Its only per-photo request is the public poses, which
  // this server does not have (404): the card stays hidden.
  const leaked = {
    id: 'p-secret',
    user_id: 'leaky-user',
    sector: 0,
    bearing_deg: 1.5,
    distance_m: 41.7,
    heading_deg: 183,
    estimated_heading_deg: 181.4,
    estimated_enu: [12.3, 45.6, 1.6],
    uploaded_at: 5,
  };
  const models = [{ ...model('m0', 'face', 0), cameras_url: '/api/models/m0/cameras' }, model('m1', 'face', 1)];
  const calls = await openLandmark({ '/api/landmarks/lm': landmarkDetail({ photos: [leaked], models }) });
  assert.deepEqual(
    calls.map((c) => c.url),
    ['/api/landmarks/lm', '/api/landmarks/lm/poses'],
  );
  const ids = doc.body.children.map((n) => n.getAttribute('id'));
  for (const id of ['photos-title', 'photo-rows', 'show-cameras']) assert.ok(!ids.includes(id), id);
  assert.equal(doc.getElementById('poses-card').hidden, true);
  assert.equal(doc.getElementById('show-poses-label').hidden, true);
  const page = text(doc.body);
  for (const s of ['p-secret', 'leaky-user', '12.3', '45.6', '41.7', '181.4', '照片方位']) assert.ok(!page.includes(s), s);
});

test('landmark page: once the 3D viewer failed to start, picking another model keeps saying so', async () => {
  // Under node, import('./viewer.js') fails like a blocked CDN does in a browser: 'three' cannot be resolved.
  await openLandmark({ '/api/landmarks/lm': landmarkDetail() });
  const message = doc.getElementById('viewer-message');
  assert.match(message.textContent, /3D 查看器无法启动/);
  doc.getElementById('model-list').querySelectorAll('.model-item')[1].click();
  await settle();
  assert.match(message.textContent, /3D 查看器无法启动/);
  assert.equal(message.hidden, false);
});

test('landmark page: a rejected HEIC upload says the format is the problem', async () => {
  globalThis.localStorage = registeredStorage();
  await openLandmark({
    '/api/landmarks/lm': landmarkDetail(),
    '/api/photos': {
      results: [
        { filename: 'IMG_1.HEIC', photo_id: null, accepted: false, reasons: ['unreadable_image'], warnings: [], landmark_id: null, sector: null, bearing_deg: null, distance_m: null },
      ],
      triggered_jobs: [],
    },
  });
  const form = doc.getElementById('upload-form');
  const button = new FakeElement('button');
  button.setAttribute('type', 'submit');
  form.append(button);
  doc.getElementById('files').files = [new File(['x'], 'IMG_1.HEIC', { type: 'image/heic' })];
  form.dispatch('submit');
  await settle();
  const [file, result] = doc.getElementById('upload-results').querySelectorAll('tbody td');
  assert.equal(text(file), 'IMG_1.HEIC');
  assert.match(text(result), /^未通过暂不支持 HEIC 格式，请导出为 JPEG/);
});

test('landmark page: while jobs run it polls the job endpoint and fetches details only on a change', async () => {
  mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_000_000 });
  let running = job({ id: 'jr', kind: 'face', sector: 2, status: 'running', finished_at: null });
  const queued = job({ id: 'jq', kind: 'face', sector: 3, status: 'queued', started_at: null, finished_at: null });
  const detailNow = () => landmarkDetail({ jobs: [queued, running] });
  const calls = await openLandmark({
    '/api/landmarks/lm': detailNow,
    '/api/jobs/jr': () => running,
    '/api/jobs/jq': () => queued,
  });
  const count = (path) => calls.filter((c) => c.url === path).length;
  assert.equal(count('/api/landmarks/lm'), 1);
  assert.equal(text(doc.getElementById('poll-state')), '自动刷新中');

  for (let i = 0; i < 3; i++) {
    mock.timers.tick(3000);
    await settle();
  }
  assert.equal(count('/api/landmarks/lm'), 1, 'no full details while nothing changed');
  assert.equal(count('/api/jobs/jr'), 3);
  assert.equal(count('/api/jobs/jq'), 3);

  running = { ...running, status: 'done', finished_at: 13 };
  mock.timers.tick(3000);
  await settle();
  assert.equal(count('/api/landmarks/lm'), 2, 'a status change fetches the details');

  // Only a queued job left: slower polling.
  const before = count('/api/jobs/jq');
  mock.timers.tick(3000);
  await settle();
  assert.equal(count('/api/jobs/jq'), before);
  mock.timers.tick(7000);
  await settle();
  assert.equal(count('/api/jobs/jq'), before + 1);

  // Other visitors' uploads still show up: full details at least every 30 s while jobs are active.
  for (let i = 0; i < 3; i++) {
    mock.timers.tick(10000);
    await settle();
  }
  assert.equal(count('/api/landmarks/lm'), 3);
});

const SHARED_KEY = 'crowd3d.shared_location.lm';
const json = (body, status = 200) => new Response(JSON.stringify(body), { status });

function uploadResult(n, shared, token) {
  return {
    filename: `IMG_${n}.jpg`,
    photo_id: `p${n}`,
    accepted: true,
    reasons: [],
    warnings: [],
    landmark_id: 'lm',
    sector: 0,
    bearing_deg: 1.5,
    distance_m: 30,
    share_location: shared,
    public_id: shared ? `pub${n}`.padEnd(12, '0') : null,
    manage_token: shared ? token : null,
  };
}

/** Opens the landmark page with a fetch mock that records each request's init; restored is a browser that ticked the
 * agreement box (of the registration form, D33) again when it restored the form after a reload. */
async function openWithRequests(handler, { restored = true } = {}) {
  fakeTimers({ fresh: true });
  doc?.restore();
  doc = installDom({ search: '?id=lm' });
  console.warn = () => {};
  if (restored) doc.getElementById('agree-terms').checked = true;
  const requests = [];
  globalThis.fetch = async (url, init = {}) => {
    requests.push({ url, init });
    return handler(url, init);
  };
  await import(`../landmark.js?page=${++page}`);
  await settle();
  return requests;
}

/** Appends the submit button the upload form has in landmark.html (the fake DOM has no markup) and returns the form. */
function uploadForm() {
  const form = doc.getElementById('upload-form');
  const button = new FakeElement('button');
  button.setAttribute('type', 'submit');
  form.append(button);
  return form;
}

/** What the upload card shows (D33): 'register', 'accept' or 'ready' (web/landmark.js registrationMode). */
function cardMode() {
  const $ = (id) => doc.getElementById(id);
  const shown = (id) => !$(id).hidden;
  if (shown('upload-form')) {
    assert.ok(shown('registered') && !shown('register-form'), 'ready: who uploads and the upload form only');
    assert.match(text($('upload-as')), /^(上传者昵称：|Uploading as )/);
    return 'ready';
  }
  assert.ok(shown('register-form'), 'the registration form');
  if (shown('nickname-field')) {
    assert.ok(!shown('registered'), 'register: the registration form only');
    return 'register';
  }
  // accept: who is registered and the way to forget it stay, above the new agreement version to accept
  assert.ok(shown('registered'), 'accept: who is registered, and Forget');
  assert.match(text($('upload-as')), /^(已登记的昵称：|Registered as )/);
  return 'accept';
}

test('landmark page: register once (a nickname and the agreement, the box unticked on every load), then each upload sends the contributor token; a photo can be withdrawn (D33)', async () => {
  const storage = fakeStorage();
  globalThis.localStorage = storage;
  const token = 'b'.repeat(64);
  const contributorToken = 'a7'.repeat(32);
  let n = 0;
  const handler = (url, init) => {
    if (url === '/api/landmarks/lm') return json(landmarkDetail({ contributors: [{ user_id: 'u' }, { user_id: 'amy' }] }));
    if (url === '/api/contributors') {
      const body = JSON.parse(init.body);
      return json({ nickname: body.nickname.trim(), terms: body.terms, contributor_token: contributorToken }, 201);
    }
    if (url === '/api/photos') {
      return json({ results: [uploadResult(++n, init.body.get('contributor_token') === contributorToken, token)], triggered_jobs: [] });
    }
    if (url === '/api/photos/p1/share_location') return json({ photo_id: 'p1', share_location: false });
    return json({ detail: 'Not Found' }, 404);
  };
  let requests = await openWithRequests(handler);
  const $ = (id) => doc.getElementById(id);
  const box = () => $('agree-terms');
  const status = () => text($('upload-status'));
  const me = () => $('contributors').querySelectorAll('li').map(text);
  // not registered: the registration form only, its box unticked although the browser restored it
  assert.equal(cardMode(), 'register');
  assert.equal(box().checked, false);
  assert.equal(text($('register-intro')), '第一次上传前，先登记昵称并同意《用户协议》。只需登记一次，这台设备会记住。');
  assert.equal(text($('register-submit')), '登记');
  assert.equal($('nickname').value, '');
  assert.deepEqual(me(), ['amy', 'u'], 'nobody is "you" before registering');
  assert.equal($('shared-own').hidden, true);

  const registers = () => requests.filter((r) => r.url === '/api/contributors');
  const uploads = () => requests.filter((r) => r.url === '/api/photos');
  const registerForm = $('register-form');
  // unticked: nothing is sent, the message says why, and the box gets the focus
  let focused = 0;
  box().focus = () => focused++;
  $('nickname').value = '  u ';
  registerForm.dispatch('submit');
  await settle();
  assert.equal(registers().length, 0);
  assert.equal(status(), '请先阅读并同意《用户协议》');
  assert.equal(focused, 1);
  // no nickname (or a control character in it): nothing is sent either
  box().checked = true;
  for (const nickname of ['  ', 'a\u0007b']) {
    $('nickname').value = nickname;
    registerForm.dispatch('submit');
    await settle();
    assert.equal(registers().length, 0);
    assert.equal(status(), '请填写昵称（不超过 64 个字符）');
  }

  $('nickname').value = '  u ';
  registerForm.dispatch('submit');
  await settle();
  assert.equal(registers().length, 1);
  assert.equal(registers()[0].init.method, 'POST');
  assert.deepEqual(JSON.parse(registers()[0].init.body), { nickname: 'u', terms: 'terms-v1' });
  assert.deepEqual(JSON.parse(storage.getItem(CONTRIBUTOR_KEY)), { token: contributorToken, nickname: 'u', terms: 'terms-v1' });
  assert.equal(status(), '已登记，现在可以上传照片了。');
  assert.equal(cardMode(), 'ready');
  assert.equal(text($('upload-as')), '上传者昵称：u');
  assert.deepEqual(me(), ['amy', 'u（你）'], 'the registered nickname is "you"');

  // uploads: the contributor token, no nickname and no agreement field, and no box to tick
  const form = uploadForm();
  const submit = async () => {
    $('files').files = [new File(['x'], 'IMG.jpg', { type: 'image/jpeg' })];
    form.dispatch('submit');
    await settle();
  };
  await submit();
  const body = uploads().at(-1).init.body;
  assert.deepEqual(body.getAll('contributor_token'), [contributorToken]);
  for (const field of ['user_id', 'terms', 'share_location']) assert.equal(body.has(field), false, field);
  // public once a build has determined its position and heading, at the earliest after the next one (D33)
  assert.match(text($('upload-results')), /通过确定了位置和朝向后公开，最早在下次建模结束后（可撤回）/);
  assert.deepEqual(JSON.parse(storage.getItem(SHARED_KEY)), [
    { photo_id: 'p1', token, filename: 'IMG_1.jpg', public_id: 'pub100000000' },
  ]);
  let own = $('shared-own');
  assert.equal(own.hidden, false);
  assert.match(text(own), /你在这台设备上公开了 1 张照片。撤回后，别人就看不到这些照片和它们的拍摄位置了。/);
  assert.match(text(own), /IMG_1\.jpg/);
  await submit();
  assert.equal(uploads().length, 2);
  assert.deepEqual(uploads().at(-1).init.body.getAll('contributor_token'), [contributorToken]);
  assert.equal(registers().length, 1, 'registered once');
  // the token only ever in a POST body: no URL holds it
  assert.ok(requests.every((r) => !r.url.includes(contributorToken)));

  // A reload: still registered, the box unticked; the list of shared photos comes back from this device
  requests = await openWithRequests(handler);
  storage.setItem(SHARED_KEY, JSON.stringify([{ photo_id: 'p1', token, filename: 'IMG_1.jpg', public_id: 'pub100000000' }]));
  requests = await openWithRequests(handler);
  assert.equal(cardMode(), 'ready');
  assert.equal(box().checked, false);
  assert.equal(text($('upload-as')), '上传者昵称：u');
  assert.equal(registers().length, 0);
  own = $('shared-own');
  assert.equal(own.hidden, false);
  const withdraw = own.querySelectorAll('button').find((b) => b.textContent === '撤回');
  assert.equal(own.querySelectorAll('button').some((b) => b.textContent === '全部撤回'), false, 'one photo: one button');
  withdraw.click();
  await settle();
  const post = requests.find((r) => r.url === '/api/photos/p1/share_location');
  assert.equal(post.init.method, 'POST');
  assert.deepEqual(JSON.parse(post.init.body), { share: false, token });
  assert.equal(storage.getItem(SHARED_KEY), null);
  assert.equal(text(own), '已撤回 1 张照片。');
});

test('landmark page: an upload refused for the registration: register_required asks to register again; terms_outdated to accept the new version once, then the same photos upload (D33)', async () => {
  const storage = registeredStorage();
  globalThis.localStorage = storage;
  const fresh = 'e5'.repeat(32);
  let refusal = null; // the 422 item /api/photos answers with
  const handler = (url, init) => {
    if (url === '/api/landmarks/lm') return json(landmarkDetail());
    if (url === '/api/contributors') return json({ nickname: JSON.parse(init.body).nickname, terms: 'terms-v1', contributor_token: fresh }, 201);
    if (url === '/api/contributors/accept') return json({ nickname: 'u', terms: 'terms-v1' });
    if (url === '/api/photos') {
      if (refusal) return json({ detail: [{ msg: 'x', input: null, ...refusal }] }, 422);
      return json({ results: [uploadResult(1, true, 'b'.repeat(64))], triggered_jobs: [] });
    }
    return json({ detail: 'Not Found' }, 404);
  };
  const requests = await openWithRequests(handler, { restored: false });
  const $ = (id) => doc.getElementById(id);
  const status = () => text($('upload-status'));
  const stored = () => JSON.parse(storage.getItem(CONTRIBUTOR_KEY));
  const form = uploadForm();
  const sent = (path) => requests.filter((r) => r.url === path);
  // the results table of the last upload: its rows (photo, result)
  const results = () => $('upload-results').querySelectorAll('tbody tr').map((tr) => tr.querySelectorAll('td').map(text));
  const uploaded = async () => {
    refusal = null;
    $('files').files = [new File(['x'], 'IMG_1.jpg', { type: 'image/jpeg' })];
    form.dispatch('submit');
    await settle();
    assert.equal(status(), '1 张通过。');
    assert.equal(results().length, 1);
    assert.match(results()[0][1], /^通过确定了位置和朝向后公开/);
  };
  assert.equal(cardMode(), 'ready');
  await uploaded();

  // register_required: the server does not know this registration; register again, the nickname filled in. The
  // results of the earlier upload go with it: nothing of them stays under the new status (D33, a screenshot round)
  refusal = { type: 'register_required', loc: ['body', 'contributor_token'] };
  $('files').files = [new File(['x'], 'IMG_0.jpg', { type: 'image/jpeg' })];
  form.dispatch('submit');
  await settle();
  assert.equal(status(), '服务器上找不到这台设备的登记，请重新登记。');
  assert.deepEqual(results(), []);
  assert.equal($('upload-results').hasChildNodes(), false);
  assert.equal(storage.getItem(CONTRIBUTOR_KEY), null);
  assert.equal(cardMode(), 'register');
  assert.equal($('nickname').value, 'u');
  assert.equal($('agree-terms').checked, false);
  assert.deepEqual(sent('/api/photos').at(-1).init.body.getAll('contributor_token'), [TOKEN]);
  $('agree-terms').checked = true;
  $('register-form').dispatch('submit');
  await settle();
  assert.deepEqual(stored(), { token: fresh, nickname: 'u', terms: 'terms-v1' });
  assert.equal(cardMode(), 'ready');
  await uploaded();

  // terms_outdated: accept the new version (no new nickname); the chosen photos stay, the earlier results go
  refusal = { type: 'terms_outdated', loc: ['body', 'contributor_token'], ctx: { version: 'terms-v1', url: '/terms.html' } };
  $('files').files = [new File(['x'], 'IMG_A.jpg', { type: 'image/jpeg' }), new File(['x'], 'IMG_B.jpg', { type: 'image/jpeg' })];
  $('files').dispatch('change');
  form.dispatch('submit');
  await settle();
  assert.equal(status(), '还没有上传：先同意新版《用户协议》，选好的照片会保留。', 'not the intro again');
  assert.equal($('upload-results').hasChildNodes(), false);
  assert.deepEqual(stored(), { token: fresh, nickname: 'u', terms: null });
  assert.equal(cardMode(), 'accept');
  assert.equal(text($('register-intro')), '《用户协议》已更新。请阅读并同意新版本后继续上传，昵称不变。');
  assert.equal(text($('register-submit')), '同意新版本');
  assert.equal($('agree-terms').checked, false);
  assert.equal(text($('files-chosen')), '已选 2 张');
  // unticked: no request
  $('register-form').dispatch('submit');
  await settle();
  assert.equal(sent('/api/contributors/accept').length, 0);
  assert.equal(status(), '请先阅读并同意《用户协议》');
  $('agree-terms').checked = true;
  const asked = sent('/api/contributors/photos').length;
  $('register-form').dispatch('submit');
  await settle();
  assert.equal(sent('/api/contributors/accept').length, 1);
  assert.deepEqual(JSON.parse(sent('/api/contributors/accept')[0].init.body), { contributor_token: fresh, terms: 'terms-v1' });
  // and the registration's photos are asked for again (D34)
  assert.equal(sent('/api/contributors/photos').length, asked + 1);
  assert.deepEqual(JSON.parse(sent('/api/contributors/photos').at(-1).init.body), { contributor_token: fresh, landmark_id: 'lm' });
  assert.equal(sent('/api/contributors').length, 1, 'no new registration');
  assert.equal(status(), '已同意新版《用户协议》，可以继续上传了。');
  assert.deepEqual(stored(), { token: fresh, nickname: 'u', terms: 'terms-v1' });
  assert.equal(cardMode(), 'ready');
  assert.equal(text($('upload-as')), '上传者昵称：u');
  // the same photos, with the same token
  refusal = null;
  form.dispatch('submit');
  await settle();
  const upload = sent('/api/photos').at(-1).init.body;
  assert.deepEqual(upload.getAll('files').map((f) => f.name), ['IMG_A.jpg', 'IMG_B.jpg']);
  assert.deepEqual(upload.getAll('contributor_token'), [fresh]);
  assert.equal(status(), '1 张通过。');
  assert.equal(results().length, 1);

  // any other refusal: the upload failed, the registration stays; the table held the results of the upload before,
  // so it is emptied when this one starts
  refusal = { type: 'missing', loc: ['body', 'files'] };
  $('files').files = [new File(['x'], 'IMG_C.jpg', { type: 'image/jpeg' })];
  form.dispatch('submit');
  await settle();
  assert.equal(status(), '上传失败：照片：必填');
  assert.deepEqual(results(), []);
  assert.equal(cardMode(), 'ready');
  assert.deepEqual(stored(), { token: fresh, nickname: 'u', terms: 'terms-v1' });
  assert.equal($('shared-own').hidden, false);
});

test("landmark page: forgetting this device's registration keeps the withdrawal list; a stored older agreement version asks to accept it on load (D33)", async () => {
  const shared = [{ photo_id: 'p1', token: 'b'.repeat(64), filename: 'IMG_1.jpg', public_id: 'pub100000000' }];
  const storage = registeredStorage('zoe', { [SHARED_KEY]: JSON.stringify(shared), 'crowd3d.lang': 'zh' });
  globalThis.localStorage = storage;
  const handler = (url) => {
    if (url === '/api/landmarks/lm') return json(landmarkDetail({ contributors: [{ user_id: 'zoe' }] }));
    if (url === '/api/photos') return json({ results: [uploadResult(2, true, 'c'.repeat(64))], triggered_jobs: [] });
    return json({ detail: 'Not Found' }, 404);
  };
  let requests = await openWithRequests(handler, { restored: false });
  const $ = (id) => doc.getElementById(id);
  assert.equal(cardMode(), 'ready');
  assert.equal(text($('upload-as')), '上传者昵称：zoe');
  assert.deepEqual($('contributors').querySelectorAll('li').map(text), ['zoe（你）']);
  // an upload first: its results show under the form
  $('files').files = [new File(['x'], 'IMG_2.jpg', { type: 'image/jpeg' })];
  uploadForm().dispatch('submit');
  await settle();
  assert.equal(text($('upload-status')), '1 张通过。');
  assert.equal($('upload-results').querySelectorAll('tbody tr').length, 1);
  const withdrawable = storage.getItem(SHARED_KEY);
  assert.deepEqual(JSON.parse(withdrawable).map((s) => s.photo_id), ['p1', 'p2']);
  // the nickname box (hidden while registered) still holds the nickname typed when registering on this page
  $('nickname').value = 'zoe';
  let focused = 0;
  $('nickname').focus = () => focused++;
  const before = requests.length;
  $('forget-registration').click();
  await settle();
  assert.equal(requests.length, before, 'forgetting is local: no request');
  assert.equal(storage.getItem(CONTRIBUTOR_KEY), null);
  assert.equal(storage.getItem(SHARED_KEY), withdrawable, 'the withdrawal list stays');
  assert.deepEqual(JSON.parse(withdrawable)[0], shared[0]);
  assert.equal(storage.getItem('crowd3d.lang'), 'zh', 'only the registration goes');
  assert.equal(cardMode(), 'register');
  assert.equal($('nickname').value, '', 'the next registration starts empty');
  assert.equal($('upload-results').hasChildNodes(), false, 'the results of the upload before go with the registration');
  assert.equal(focused, 1);
  assert.equal(text($('upload-status')), '已忘记这台设备上的登记。在这台设备上公开的照片仍可以在下面撤回。');
  assert.deepEqual($('contributors').querySelectorAll('li').map(text), ['zoe'], 'nobody is "you" any more');
  const own = $('shared-own');
  assert.equal(own.hidden, false);
  assert.ok(own.querySelectorAll('button').some((b) => b.textContent === '撤回'));

  // no photos shared from this device here: nothing to point to below
  storage.setItem(CONTRIBUTOR_KEY, JSON.stringify({ token: TOKEN, nickname: 'zoe', terms: 'terms-v1' }));
  storage.removeItem(SHARED_KEY);
  await openWithRequests(handler);
  $('forget-registration').click();
  await settle();
  assert.equal(text($('upload-status')), '已忘记这台设备上的登记。');
  assert.equal(cardMode(), 'register');
  inLang('en', () => assert.equal(t('register.forgottenPlain'), "This device's registration is forgotten."));

  // a registration under an older agreement version: accept the new one right away, nothing else to fill in
  storage.setItem(CONTRIBUTOR_KEY, JSON.stringify({ token: TOKEN, nickname: 'zoe', terms: 'terms-v0' }));
  requests = await openWithRequests(handler);
  assert.equal(cardMode(), 'accept');
  assert.equal($('agree-terms').checked, false, 'unticked although the browser restored it');
  assert.equal(text($('register-intro')), '《用户协议》已更新。请阅读并同意新版本后继续上传，昵称不变。');
  assert.equal(text($('register-submit')), '同意新版本');
  assert.deepEqual($('contributors').querySelectorAll('li').map(text), ['zoe（你）']);
  // nothing registered or accepted on load; the registration's photos are asked for whatever version it accepted (D34)
  assert.deepEqual(requests.filter((r) => r.url.startsWith('/api/contributors')).map((r) => r.url), ['/api/contributors/photos']);
  assert.equal(text($('upload-as')), '已登记的昵称：zoe');
  inLang('en', () => assert.equal(t('register.as', { name: 'zoe' }), 'Registered as zoe'));
  // one who does not want the new version can forget the registration here too
  $('forget-registration').click();
  await settle();
  assert.equal(cardMode(), 'register');
  assert.equal(storage.getItem(CONTRIBUTOR_KEY), null);
  assert.equal(text($('upload-status')), '已忘记这台设备上的登记。');
  // a damaged entry is no registration
  for (const bad of ['{', '"x"', JSON.stringify({ token: '', nickname: 'zoe', terms: 'terms-v1' }), JSON.stringify({ token: TOKEN, terms: 'terms-v1' })]) {
    storage.setItem(CONTRIBUTOR_KEY, bad);
    await openWithRequests(handler);
    assert.equal(cardMode(), 'register', bad);
  }
});

test('landmark page: with storage blocked the registration lasts for the page view; the nickname D32 remembered fills in the registration and is removed once registered; refused registrations (D33)', async () => {
  const blocked = {
    getItem() {
      throw new DOMException('blocked', 'SecurityError');
    },
    setItem() {
      throw new DOMException('blocked', 'SecurityError');
    },
    removeItem() {
      throw new DOMException('blocked', 'SecurityError');
    },
  };
  globalThis.localStorage = blocked;
  const contributorToken = 'd4'.repeat(32);
  let registerAnswer = null; // a function of (init) -> Response, or null for success
  const handler = (url, init) => {
    if (url === '/api/landmarks/lm') return json(landmarkDetail({ contributors: [{ user_id: 'kim' }] }));
    if (url === '/api/contributors') {
      if (registerAnswer) return registerAnswer(init);
      return json({ nickname: JSON.parse(init.body).nickname, terms: 'terms-v1', contributor_token: contributorToken }, 201);
    }
    if (url === '/api/contributors/accept') return json({ detail: [{ type: 'register_required', loc: ['body', 'contributor_token'], msg: 'x', input: null }] }, 422);
    if (url === '/api/photos') return json({ results: [uploadResult(1, true, 'b'.repeat(64))], triggered_jobs: [] });
    return json({ detail: 'Not Found' }, 404);
  };
  let requests = await openWithRequests(handler, { restored: false });
  const $ = (id) => doc.getElementById(id);
  const status = () => text($('upload-status'));
  assert.equal(cardMode(), 'register');
  $('nickname').value = 'kim';
  $('agree-terms').checked = true;

  // refused: a newer agreement than this page (reload it), a nickname the server refuses, no answer
  for (const [answer, message] of [
    [() => json({ detail: [{ type: 'terms_outdated', loc: ['body', 'terms'], msg: 'x', input: 'terms-v1', ctx: { version: 'terms-v2' } }] }, 422),
      '用户协议已更新，请刷新页面后重新阅读并同意'],
    [() => json({ detail: [{ type: 'value_error', loc: ['body', 'nickname'], msg: 'x', input: 'kim' }] }, 422), '登记失败：昵称：取值无效'],
    [() => { throw new TypeError('Failed to fetch'); }, '登记失败：无法连接服务器，请检查网络后重试'],
    // an answer without a token stores nothing
    [() => json({ nickname: 'kim', terms: 'terms-v1' }, 201), '登记失败：请求失败（HTTP 201）'],
  ]) {
    registerAnswer = answer;
    $('register-form').dispatch('submit');
    await settle();
    assert.equal(status(), message);
    assert.equal(cardMode(), 'register');
    assert.equal($('agree-terms').checked, true, 'still ticked for another try');
    assert.equal($('register-submit').disabled, false);
    assert.equal(text($('register-submit')), '登记');
  }
  registerAnswer = null;
  $('register-form').dispatch('submit');
  await settle();
  // registered for this page view, and the page says that it does not last
  assert.equal(status(), '已登记，现在可以上传照片了。这台设备不能保存登记，刷新后需要重新登记。');
  inLang('en', () => assert.equal(
    t('register.done') + t('common.sentenceSep') + t('register.notKept'),
    'Registered. You can upload photos now. This device cannot keep the registration; you will need to register again after reloading.',
  ));
  assert.equal(cardMode(), 'ready');
  assert.equal(text($('upload-as')), '上传者昵称：kim');
  assert.deepEqual($('contributors').querySelectorAll('li').map(text), ['kim（你）']);
  const form = uploadForm();
  $('files').files = [new File(['x'], 'IMG.jpg', { type: 'image/jpeg' })];
  form.dispatch('submit');
  await settle();
  assert.deepEqual(requests.filter((r) => r.url === '/api/photos').at(-1).init.body.getAll('contributor_token'), [contributorToken]);
  assert.equal(status(), '1 张通过。');
  // nothing kept: a reload asks again
  requests = await openWithRequests(handler, { restored: false });
  assert.equal(cardMode(), 'register');

  // the nickname D32 remembered: fills in the registration, and is removed once registered
  const storage = fakeStorage({ 'crowd3d.user_id': 'old name' });
  globalThis.localStorage = storage;
  requests = await openWithRequests(handler, { restored: false });
  assert.equal($('nickname').value, 'old name');
  $('agree-terms').checked = true;
  $('register-form').dispatch('submit');
  await settle();
  assert.deepEqual(JSON.parse(requests.find((r) => r.url === '/api/contributors').init.body), { nickname: 'old name', terms: 'terms-v1' });
  assert.equal(storage.getItem('crowd3d.user_id'), null);
  assert.deepEqual(JSON.parse(storage.getItem(CONTRIBUTOR_KEY)), { token: contributorToken, nickname: 'old name', terms: 'terms-v1' });
  requests = await openWithRequests(handler, { restored: false });
  assert.equal(cardMode(), 'ready');
  assert.equal($('nickname').value, '');

  // accepting for a registration the server no longer knows: register again, the nickname filled in
  storage.setItem(CONTRIBUTOR_KEY, JSON.stringify({ token: contributorToken, nickname: 'old name', terms: null }));
  requests = await openWithRequests(handler, { restored: false });
  assert.equal(cardMode(), 'accept');
  $('agree-terms').checked = true;
  $('register-form').dispatch('submit');
  await settle();
  assert.equal(status(), '服务器上找不到这台设备的登记，请重新登记。');
  assert.equal(cardMode(), 'register');
  assert.equal($('nickname').value, 'old name');
  assert.equal(storage.getItem(CONTRIBUTOR_KEY), null);
});

test('landmark page: withdrawing all drops only the photos withdrawn or gone; a refused token stays on the list (D30)', async () => {
  const shared = ['p1', 'p2', 'p3', 'p4'].map((id) => ({ photo_id: id, token: id.repeat(32).slice(0, 64), filename: `${id}.jpg` }));
  const storage = fakeStorage({ [SHARED_KEY]: JSON.stringify([...shared, { photo_id: 7 }, null]) });
  globalThis.localStorage = storage;
  await openWithRequests(
    (url) => {
      if (url === '/api/landmarks/lm') return json(landmarkDetail());
      if (url === '/api/photos/p1/share_location') return json({ photo_id: 'p1', share_location: false });
      // p2: the server refuses the token (another salt): the photo may still be shared
      if (url === '/api/photos/p2/share_location') return json({ detail: 'wrong manage token' }, 403);
      if (url === '/api/photos/p4/share_location') return json({ detail: 'unknown photo' }, 404);
      throw new TypeError('Failed to fetch'); // p3: the network is down
    },
    { restored: false },
  );
  const own = doc.getElementById('shared-own');
  assert.match(text(own), /你在这台设备上公开了 4 张照片。/);       // malformed entries are dropped
  assert.equal(own.querySelector('ul').getAttribute('aria-label'), '公开的照片');
  own.querySelectorAll('button').find((b) => b.textContent === '全部撤回').click();
  await settle();
  assert.deepEqual(JSON.parse(storage.getItem(SHARED_KEY)), [shared[1], shared[2]]);
  assert.match(
    text(own),
    /已撤回 1 张照片；1 张服务器已找不到，无法撤回；1 张的撤回凭证已失效，照片和位置可能仍然公开；1 张撤回失败，请稍后再试。$/,
  );
  assert.match(text(own), /公开了 2 张照片。/);
  assert.equal(own.querySelectorAll('button').filter((b) => b.textContent === '撤回').length, 2);
});

test('landmark page: a photo withdrawn right after its upload no longer reads as public in the upload results; one the server no longer has reads as not public; a refused one stays public (D33, Chinese and English)', async () => {
  const want = {
    zh: { public: '确定了位置和朝向后公开，最早在下次建模结束后（可撤回）', withdrawn: '已撤回，不再公开', gone: '未公开', one: '撤回', all: '全部撤回' },
    en: {
      public: 'Public once its position and heading are known, at the earliest after the next build (you can withdraw it)',
      withdrawn: 'Withdrawn: no longer public',
      gone: 'Not public',
      one: 'Withdraw',
      all: 'Withdraw all',
    },
  };
  for (const lang of ['zh', 'en']) {
    await inLang(lang, async () => {
      const w = want[lang];
      assert.equal(t('upload.withdrawn'), w.withdrawn);
      globalThis.localStorage = registeredStorage();
      const handler = (url) => {
        if (url === '/api/landmarks/lm') return json(landmarkDetail());
        if (url === '/api/photos') {
          const rejected = { ...uploadResult(5, false, null), photo_id: null, accepted: false, reasons: ['no_gps'] };
          return json({ results: [1, 2, 3, 4].map((n) => uploadResult(n, true, `${n}`.repeat(64))).concat(rejected), triggered_jobs: [] });
        }
        if (url === '/api/photos/p1/share_location') return json({ photo_id: 'p1', share_location: false });
        if (url === '/api/photos/p2/share_location') return json({ photo_id: 'p2', share_location: false });
        if (url === '/api/photos/p3/share_location') return json({ detail: 'unknown photo' }, 404); // gone
        if (url === '/api/photos/p4/share_location') return json({ detail: 'wrong manage token' }, 403); // may still be public
        return json({ detail: 'Not Found' }, 404);
      };
      await openWithRequests(handler, { restored: false });
      const $ = (id) => doc.getElementById(id);
      const form = uploadForm();
      $('files').files = [new File(['x'], 'IMG.jpg', { type: 'image/jpeg' })];
      form.dispatch('submit');
      await settle();
      // each row's photo and what it says about being public (the rejected photo says nothing)
      const rows = () =>
        $('upload-results')
          .querySelectorAll('tbody tr')
          .map((tr) => [text(tr.querySelector('.cell-file span')), tr.querySelector('.share-state') ? text(tr.querySelector('.share-state')) : null]);
      assert.deepEqual(rows(), [['IMG_1.jpg', w.public], ['IMG_2.jpg', w.public], ['IMG_3.jpg', w.public], ['IMG_4.jpg', w.public], ['IMG_5.jpg', null]]);
      const own = $('shared-own');
      const buttons = () => own.querySelectorAll('button');
      // one photo: its row only
      buttons().filter((b) => text(b) === w.one)[0].click();
      await settle();
      assert.deepEqual(rows(), [['IMG_1.jpg', w.withdrawn], ['IMG_2.jpg', w.public], ['IMG_3.jpg', w.public], ['IMG_4.jpg', w.public], ['IMG_5.jpg', null]]);
      // all the others: withdrawn, gone (not public) and refused (still on the list, so still public)
      buttons().find((b) => text(b) === w.all).click();
      await settle();
      assert.deepEqual(rows(), [['IMG_1.jpg', w.withdrawn], ['IMG_2.jpg', w.withdrawn], ['IMG_3.jpg', w.gone], ['IMG_4.jpg', w.public], ['IMG_5.jpg', null]]);
      assert.equal(buttons().filter((b) => text(b) === w.one).length, 1, 'the refused photo stays on the list');
      // no row of a photo that left the list says it is (or will be) public
      const listed = JSON.parse(globalThis.localStorage.getItem(SHARED_KEY)).map((s) => s.photo_id);
      assert.deepEqual(listed, ['p4']);
      for (const [file, state] of rows()) if (file !== 'IMG_4.jpg') assert.notEqual(state, w.public, file);
    });
  }
});

// ---------- the list of shared photos: several tabs, a browser that does not keep it, recovery from the server (D34) ----------

/**
 * A browser with several tabs of the landmark page: one localStorage (map: what it holds) and one fetch for all, a
 * document and a page script instance per tab. A tab's code runs with its document current (act). What one tab
 * writes reaches every other tab as a storage event, delivered when the test next acts in that tab, as a browser
 * delivers it before the tab's next task, or later ({deliver: false}: the tab has not heard yet; the events wait).
 * refuse: keys the storage does not take (setItem throws, as a full or write-blocked storage does).
 */
function fakeBrowser(handler, initial = {}, { refuse = () => false } = {}) {
  const map = new Map(Object.entries(initial));
  const tabs = [];
  const requests = [];
  let current = null;
  const notify = (key, oldValue, newValue) => {
    for (const tab of tabs) if (tab !== current) tab.pending.push({ key, oldValue, newValue });
  };
  globalThis.localStorage = {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => {
      if (refuse(k)) throw new DOMException('quota', 'QuotaExceededError');
      const old = map.has(k) ? map.get(k) : null;
      map.set(k, String(v));
      if (old !== String(v)) notify(k, old, String(v));
    },
    removeItem: (k) => {
      if (!map.has(k)) return;
      const old = map.get(k);
      map.delete(k);
      notify(k, old, null);
    },
  };
  globalThis.fetch = async (url, init = {}) => {
    requests.push({ url, init, tab: current });
    return handler(url, init);
  };
  console.warn = () => {};
  return {
    map,
    requests,
    /** One more tab of the landmark page. */
    async open() {
      fakeTimers();
      const tab = { doc: installDom({ search: '?id=lm' }), pending: [] };
      tabs.push(tab);
      current = tab;
      await import(`../landmark.js?page=${++page}`);
      await settle();
      return tab;
    },
    /** Acts in a tab: first the storage events it has not had yet (unless deliver is false: they keep waiting). */
    async act(tab, fn = () => {}, { deliver = true } = {}) {
      useDom(tab.doc);
      current = tab;
      if (deliver) for (const event of tab.pending.splice(0)) tab.doc.dispatchWindow('storage', event);
      await settle();
      const result = await fn(tab.doc);
      await settle();
      return result;
    },
    close() {
      for (const tab of [...tabs].reverse()) tab.doc.restore();
    },
  };
}

/** Uploads one file from the page in d (the submit button of landmark.html appended once). */
function submitUpload(d) {
  const form = d.getElementById('upload-form');
  if (!form.querySelector('button')) {
    const button = new FakeElement('button');
    button.setAttribute('type', 'submit');
    form.append(button);
  }
  d.getElementById('files').files = [new File(['x'], 'IMG.jpg', { type: 'image/jpeg' })];
  form.dispatch('submit');
}

// The pauses before the page asks again for the registration's photos (web/landmark.js RECOVER_AGAIN_MS, D34)
const RECOVER_PAUSES = [3000, 8000, 20000, 45000, 90000];

/** Lets ms pass on the page's (fake) clock, then the answers to what its timers asked arrive. */
async function pass(ms) {
  mock.timers.tick(ms);
  await settle();
}

/** Lets every pause of the asks under way pass: after it none is. */
async function passRecoverPauses() {
  for (const ms of RECOVER_PAUSES) await pass(ms);
}

/** The names the list of shared photos shows in d, in order. */
const sharedNames = (d) => d.getElementById('shared-own').querySelectorAll('.shared-name').map(text);

/** Clicks the nth "withdraw" button of the list in d (label: the button's text in the page language). */
function clickWithdraw(d, n, label = '撤回') {
  d.getElementById('shared-own').querySelectorAll('button').filter((b) => text(b) === label)[n].click();
}

test("landmark page: two tabs of one landmark keep one list of shared photos: a change is made on what the browser holds at that moment, not on a tab's older copy, and each tab follows the other (D34)", async () => {
  let n = 0; // the server numbers the photos it accepts
  let uploadFails = false; // the next uploads get no answer
  let listPhotos = null; // what the server answers the registration's list with; null: no answer
  const handler = (url) => {
    if (url === '/api/landmarks/lm') return json(landmarkDetail());
    if (url === '/api/photos') {
      if (uploadFails) throw new TypeError('Failed to fetch');
      n++;
      return json({ results: [uploadResult(n, true, `${n}`.repeat(64))], triggered_jobs: [] });
    }
    const withdrawn = url.match(/^\/api\/photos\/(p\d)\/share_location$/);
    if (withdrawn) return json({ photo_id: withdrawn[1], share_location: false });
    if (url === '/api/contributors/photos' && listPhotos) return Promise.resolve(listPhotos).then((photos) => json({ photos }));
    // else the registration's photos are not answered: what the browser stores must do on its own
    return json({ detail: 'Not Found' }, 404);
  };
  const browser = fakeBrowser(handler, { [CONTRIBUTOR_KEY]: JSON.stringify({ token: TOKEN, nickname: 'u', terms: 'terms-v1' }) });
  try {
    const stored = () => JSON.parse(browser.map.get(SHARED_KEY) ?? '[]').map((s) => s.photo_id);
    const b = await browser.open(); // opened first: its copy of the list is empty
    const a = await browser.open();
    // A uploads p1. B has not heard of it and uploads p2: the review's case, which left the browser with p2's token
    // only, and p1 public with nobody able to withdraw it
    await browser.act(a, submitUpload);
    assert.deepEqual(stored(), ['p1']);
    await browser.act(b, submitUpload, { deliver: false });
    assert.deepEqual(stored(), ['p1', 'p2']);
    assert.deepEqual(JSON.parse(browser.map.get(SHARED_KEY)), [
      { photo_id: 'p1', token: '1'.repeat(64), filename: 'IMG_1.jpg', public_id: 'pub100000000' },
      { photo_id: 'p2', token: '2'.repeat(64), filename: 'IMG_2.jpg', public_id: 'pub200000000' },
    ]);
    assert.deepEqual(sharedNames(b.doc), ['IMG_1.jpg', 'IMG_2.jpg'], 'B shows the list it wrote');
    // A follows once the storage event reaches it
    assert.deepEqual(sharedNames(a.doc), ['IMG_1.jpg']);
    await browser.act(a);
    assert.deepEqual(sharedNames(a.doc), ['IMG_1.jpg', 'IMG_2.jpg']);
    assert.match(text(a.doc.getElementById('shared-own')), /^你在这台设备上公开了 2 张照片。/);

    // a withdrawal takes off only what it withdrew: A uploads p3, B (its copy still [p1, p2]) withdraws p2
    await browser.act(a, submitUpload);
    assert.deepEqual(stored(), ['p1', 'p2', 'p3']);
    await browser.act(b, (d) => clickWithdraw(d, 1), { deliver: false });
    const post = browser.requests.filter((r) => r.url.endsWith('/share_location')).at(-1);
    assert.equal(post.url, '/api/photos/p2/share_location');
    assert.deepEqual(JSON.parse(post.init.body), { share: false, token: '2'.repeat(64) });
    assert.deepEqual(stored(), ['p1', 'p3'], 'p3, which B never saw, keeps its token');
    assert.deepEqual(sharedNames(b.doc), ['IMG_1.jpg', 'IMG_3.jpg']);
    assert.match(text(b.doc.getElementById('shared-own')), /已撤回 1 张照片。$/);
    await browser.act(a);
    assert.deepEqual(sharedNames(a.doc), ['IMG_1.jpg', 'IMG_3.jpg']);

    // A's upload results still show p3 as public; B withdraws p3: A's row no longer reads as public (D33's rule,
    // across tabs), and p3 leaves A's list
    const row = () => text(a.doc.getElementById('upload-results').querySelector('.share-state'));
    assert.equal(row(), '确定了位置和朝向后公开，最早在下次建模结束后（可撤回）');
    await browser.act(b, (d) => clickWithdraw(d, 1));
    assert.deepEqual(stored(), ['p1']);
    assert.equal(row(), '确定了位置和朝向后公开，最早在下次建模结束后（可撤回）', 'until A hears of it');
    await browser.act(a);
    assert.equal(row(), '未公开');
    assert.deepEqual(sharedNames(a.doc), ['IMG_1.jpg']);

    // an event for another key changes nothing here (B's withdrawal message stays); one for the list re-reads it
    const own = b.doc.getElementById('shared-own');
    assert.match(text(own), /已撤回 1 张照片。$/);
    await browser.act(b, (d) => d.dispatchWindow('storage', { key: 'crowd3d.lang', newValue: 'en' }));
    assert.match(text(own), /已撤回 1 张照片。$/);
    await browser.act(b, (d) => d.dispatchWindow('storage', { key: SHARED_KEY }));
    assert.match(text(own), /已撤回 1 张照片。$/, 'the same list: nothing drawn again');
    // the last photo withdrawn in A: the entry goes, and B's list is empty
    await browser.act(a, (d) => clickWithdraw(d, 0));
    assert.equal(browser.map.has(SHARED_KEY), false);
    await browser.act(b);
    assert.deepEqual(sharedNames(b.doc), []);
    assert.equal(own.hidden, true);
    // an answer for the registration's photos, made before another tab withdrew one and arriving after this tab
    // heard of the withdrawal, does not bring that photo back
    await browser.act(a, submitUpload);
    await browser.act(b);
    assert.deepEqual(sharedNames(b.doc), ['IMG_4.jpg']);
    let release;
    listPhotos = new Promise((resolve) => {
      release = () => resolve([{ photo_id: 'p4', landmark_id: 'lm', public_id: 'pub400000000', manage_token: '4'.repeat(64) }]);
    });
    await browser.act(b, (d) => d.dispatch('visibilitychange')); // B asks; the answer is on its way
    await browser.act(a, (d) => clickWithdraw(d, 0));
    await browser.act(b); // B hears that p4 left the list
    assert.deepEqual(sharedNames(b.doc), []);
    release();
    await browser.act(b);
    assert.deepEqual(sharedNames(b.doc), [], 'p4 stays off the list');
    assert.equal(browser.map.has(SHARED_KEY), false);
    listPhotos = null;
    // a tab that looks at the list for another reason before its event comes (an upload that got no answer asks
    // the server, which has nothing more) shows what the browser holds, and still does once the event has come
    await browser.act(a, submitUpload);
    uploadFails = true;
    listPhotos = [];
    await browser.act(b, submitUpload, { deliver: false });
    assert.deepEqual(sharedNames(b.doc), ['IMG_5.jpg']);
    await browser.act(b);
    assert.deepEqual(sharedNames(b.doc), ['IMG_5.jpg']);
    uploadFails = false;
    listPhotos = null;
    // the storage cleared in another tab (key null): the list goes here too, and the rows of the upload results stay
    // as they are (nothing was withdrawn)
    browser.map.clear();
    await browser.act(a, (d) => d.dispatchWindow('storage', { key: null }));
    assert.deepEqual(sharedNames(a.doc), []);
    assert.equal(row(), '确定了位置和朝向后公开，最早在下次建模结束后（可撤回）');
    // no tab ever wrote its token into a URL
    assert.ok(browser.requests.every((r) => !r.url.includes(TOKEN)));
  } finally {
    browser.close();
  }
});

test('landmark page: when the browser does not keep the list of shared photos the page says what that means for withdrawing later, with and without a registration kept on this device (D34, Chinese and English)', async () => {
  const want = {
    zh: {
      notKept: '这台设备不能保存这份列表。关掉或刷新页面后，要靠这台设备上的登记才能找回并撤回这些照片；登记没有了，就撤不回了。',
      notKeptAtAll: '这台设备不能保存这份列表，也没有保存登记。关掉或刷新页面后，你就不能再自己撤回这些照片了：要撤回，请现在就撤回。',
      unnamed: '照片（没有文件名）',
      recovered: '按这台设备的登记，从服务器找回了 2 张照片（没有文件名）。',
      one: '撤回',
      withdrawn: '已撤回 1 张照片。',
    },
    en: {
      notKept: "This device cannot keep this list. After you close or reload the page, these photos can only be found and withdrawn again with this device's registration; without it, they cannot be.",
      notKeptAtAll: 'This device cannot keep this list, nor a registration. After you close or reload the page you can no longer withdraw these photos yourself: if you want to withdraw them, do it now.',
      unnamed: 'Photo (no file name)',
      recovered: "Found 2 photos on the server under this device's registration (without their file names).",
      one: 'Withdraw',
      withdrawn: 'Withdrew 1 photo.',
    },
  };
  for (const lang of ['zh', 'en']) {
    await inLang(lang, async () => {
      const w = want[lang];
      const server = []; // the photos the server holds for the registration
      let n = 0;
      const handler = (url, init) => {
        if (url === '/api/landmarks/lm') return json(landmarkDetail({ name: 'Tower' }));
        if (url === '/api/contributors') return json({ nickname: 'kim', terms: 'terms-v1', contributor_token: TOKEN }, 201);
        if (url === '/api/contributors/photos') return json({ photos: server });
        if (url === '/api/photos') {
          n++;
          const result = uploadResult(n, true, `${n}`.repeat(64));
          server.push({ photo_id: result.photo_id, landmark_id: 'lm', public_id: result.public_id, manage_token: result.manage_token });
          return json({ results: [result], triggered_jobs: [] });
        }
        const withdrawn = url.match(/^\/api\/photos\/(p\d)\/share_location$/);
        if (withdrawn && JSON.parse(init.body).token === withdrawn[1].slice(1).repeat(64)) {
          server.splice(server.findIndex((p) => p.photo_id === withdrawn[1]), 1);
          return json({ photo_id: withdrawn[1], share_location: false });
        }
        return json({ detail: 'Not Found' }, 404);
      };
      const notice = () => doc.getElementById('shared-own').querySelector('.shared-own-warning');

      // 1. storage blocked altogether: the registration lasts for the page view, and so does the list
      globalThis.localStorage = {
        getItem() {
          throw new DOMException('blocked', 'SecurityError');
        },
        setItem() {
          throw new DOMException('blocked', 'SecurityError');
        },
        removeItem() {
          throw new DOMException('blocked', 'SecurityError');
        },
      };
      await openWithRequests(handler, { restored: false });
      doc.getElementById('nickname').value = 'kim';
      doc.getElementById('agree-terms').checked = true;
      doc.getElementById('register-form').dispatch('submit');
      await settle();
      assert.equal(doc.getElementById('shared-own').hidden, true, 'nothing shared yet: nothing to say');
      submitUpload(doc);
      await settle();
      assert.deepEqual(sharedNames(doc), ['IMG_1.jpg']);
      assert.equal(text(notice()), w.notKeptAtAll);
      assert.equal(notice().getAttribute('role'), 'note');
      submitUpload(doc);
      await settle();
      assert.deepEqual(sharedNames(doc), ['IMG_1.jpg', 'IMG_2.jpg'], 'the page keeps what the browser does not');
      assert.equal(text(notice()), w.notKeptAtAll);
      // withdrawing works from the page's own copy, and takes off only that photo
      clickWithdraw(doc, 0, w.one);
      await settle();
      assert.deepEqual(sharedNames(doc), ['IMG_2.jpg']);
      assert.equal(text(notice()), w.notKeptAtAll);
      assert.equal(text(doc.getElementById('shared-own').querySelector('.form-status')), w.withdrawn);
      clickWithdraw(doc, 0, w.one);
      await settle();
      assert.equal(notice(), null, 'an empty list: nothing is lost with the page');
      assert.equal(text(doc.getElementById('shared-own')), w.withdrawn);
      if (lang === 'en') assertNoChinese(doc.body, 'blocked storage');

      // 2. a storage that keeps the registration but does not take the list (full, or the write refused)
      n = 0;
      server.length = 0;
      const refused = [];
      let takes = false; // whether the storage takes the list (again)
      const storage = registeredStorage('kim');
      const setItem = storage.setItem;
      storage.setItem = (k, v) => {
        if (k === SHARED_KEY && !takes) {
          refused.push(v);
          throw new DOMException('quota', 'QuotaExceededError');
        }
        return setItem(k, v);
      };
      globalThis.localStorage = storage;
      await openWithRequests(handler, { restored: false });
      assert.equal(doc.getElementById('shared-own').hidden, true);
      submitUpload(doc);
      await settle();
      submitUpload(doc);
      await settle();
      assert.ok(refused.length >= 2, 'the page tried to store the list, each time it changed');
      assert.deepEqual(JSON.parse(refused.at(-1)).map((s) => s.photo_id), ['p1', 'p2'], 'the whole list, each time');
      assert.equal(storage.getItem(SHARED_KEY), null);
      assert.deepEqual(sharedNames(doc), ['IMG_1.jpg', 'IMG_2.jpg']);
      // with the registration kept on this device, the photos can be found again: the notice says so
      assert.equal(text(notice()), w.notKept);
      // a reload: the list is gone from the browser, and comes back from the server by the registration, without
      // file names; the browser still does not keep it
      const requests = await openWithRequests(handler, { restored: false });
      const asked = requests.filter((r) => r.url === '/api/contributors/photos');
      assert.equal(asked.length, 1);
      assert.equal(asked[0].init.method, 'POST');
      assert.deepEqual(JSON.parse(asked[0].init.body), { contributor_token: TOKEN, landmark_id: 'lm' });
      assert.deepEqual(sharedNames(doc), [w.unnamed, w.unnamed]);
      assert.equal(text(doc.getElementById('shared-own').querySelector('.form-status')), w.recovered);
      assert.equal(text(notice()), w.notKept);
      // such a row is withdrawn like any other, with the manage token the server gave again
      clickWithdraw(doc, 1, w.one);
      await settle();
      const post = requests.find((r) => r.url === '/api/photos/p2/share_location');
      assert.deepEqual(JSON.parse(post.init.body), { share: false, token: '2'.repeat(64) });
      assert.deepEqual(sharedNames(doc), [w.unnamed]);
      // the storage takes the list again (room was made): the next look at the list stores it, and the notice goes
      takes = true;
      doc.dispatch('visibilitychange');
      await settle();
      assert.deepEqual(JSON.parse(storage.getItem(SHARED_KEY)).map((s) => s.photo_id), ['p1']);
      assert.equal(notice(), null);
      assert.deepEqual(sharedNames(doc), [w.unnamed]);
      // and refuses it once more: the notice is back, and the photo the browser holds stays listed with the new one
      takes = false;
      submitUpload(doc);
      await settle();
      assert.equal(text(notice()), w.notKept);
      assert.deepEqual(sharedNames(doc), [w.unnamed, 'IMG_3.jpg']);
      assert.deepEqual(JSON.parse(storage.getItem(SHARED_KEY)).map((s) => s.photo_id), ['p1'], 'what it took before');
      // forgetting the registration: nothing on this device finds these photos again once the page is gone
      doc.getElementById('forget-registration').click();
      await settle();
      assert.equal(text(notice()), w.notKeptAtAll);
      assert.deepEqual(sharedNames(doc), [w.unnamed, 'IMG_3.jpg'], 'still withdrawable now');
      if (lang === 'en') assertNoChinese(doc.body, 'a storage that does not take the list');
    });
  }
});

test("landmark page: photos whose manage tokens this device lost or never got come back from the server by its registration: after an upload whose answer never arrived, on load, and when the tab is shown again; such a row has a neutral name and is withdrawn like any other (D34, Chinese and English)", async () => {
  const want = {
    zh: {
      unnamed: '照片（没有文件名）',
      storing: (k) => `没有收到上传的结果，但服务器目前已经收下了 ${k} 张照片：它们列在下面，可以撤回。服务器可能还在处理其余的，页面会接着查。`,
      lost: (k) => `没有收到上传的结果，但服务器已经收下了 ${k} 张照片：它们列在下面，可以撤回。还没收下的可以再传一次。`,
      recovered: (k) => `按这台设备的登记，从服务器找回了 ${k} 张照片（没有文件名）。`,
      unreachable: '上传失败：无法连接服务器，请检查网络后重试',
      server: '上传失败：服务器出错了，请稍后重试（HTTP 500）',
      tooLarge: /^上传失败：/,
      shared: (k) => new RegExp(`^你在这台设备上公开了 ${k} 张照片。`),
      one: '撤回',
      mine: '你的',
    },
    en: {
      unnamed: 'Photo (no file name)',
      storing: (k) =>
        k === 1
          ? "The upload's result did not arrive, but the server has stored 1 photo so far: it is listed below and can be withdrawn. The server may still be working on the rest; this page keeps checking."
          : `The upload's result did not arrive, but the server has stored ${k} photos so far: they are listed below and can be withdrawn. The server may still be working on the rest; this page keeps checking.`,
      lost: (k) =>
        k === 1
          ? "The upload's result did not arrive, but the server did store 1 photo: it is listed below and can be withdrawn. You can upload the rest again."
          : `The upload's result did not arrive, but the server did store ${k} photos: they are listed below and can be withdrawn. You can upload the rest again.`,
      recovered: (k) =>
        k === 1
          ? "Found 1 photo on the server under this device's registration (without its file name)."
          : `Found ${k} photos on the server under this device's registration (without their file names).`,
      unreachable: 'Upload failed: Could not reach the server. Check your connection and try again.',
      server: 'Upload failed: Something went wrong on the server. Try again later (HTTP 500).',
      tooLarge: /^Upload failed: /,
      shared: (k) => new RegExp(`^You shared ${k} photos? from this device\\.`),
      one: 'Withdraw',
      mine: 'Yours',
    },
  };
  const manage = (id) => id.slice(1).repeat(64).slice(0, 64); // the manage token of photo "pN"
  const own = (id, over = {}) => ({ photo_id: id, landmark_id: 'lm', public_id: `${id}pub`.padEnd(12, '0'), manage_token: manage(id), ...over });
  for (const lang of ['zh', 'en']) {
    await inLang(lang, async () => {
      const w = want[lang];
      // what the server holds for this registration; p1 was uploaded from this device with its answer received
      let server = [own('p1')];
      let listAnswer = null; // a Response to answer the list with instead
      let uploadAnswer = null; // what the next upload does: a function -> Response (or throws)
      const handler = (url, init) => {
        if (url === '/api/landmarks/lm') return json(landmarkDetail({ name: 'Tower' }));
        if (url === '/api/landmarks/lm/poses') return json(posesDoc({ photos: [{ ...posesDoc().photos[0], id: own('p2').public_id }, posesDoc().photos[2]] }));
        if (url === '/api/contributors/photos') return listAnswer ? listAnswer() : json({ photos: server });
        if (url === '/api/photos') return uploadAnswer();
        const withdrawn = url.match(/^\/api\/photos\/(p\d)\/share_location$/);
        if (withdrawn && JSON.parse(init.body).token === manage(withdrawn[1])) {
          server = server.filter((p) => p.photo_id !== withdrawn[1]);
          return json({ photo_id: withdrawn[1], share_location: false });
        }
        return json({ detail: 'wrong token for this photo' }, 403);
      };
      const mine = { photo_id: 'p1', token: manage('p1'), filename: 'IMG_1.jpg', public_id: own('p1').public_id };
      const storage = registeredStorage('u', { [SHARED_KEY]: JSON.stringify([mine]) });
      let writes = 0; // how often the page wrote the list
      const setItem = storage.setItem;
      storage.setItem = (k, v) => {
        if (k === SHARED_KEY) writes++;
        return setItem(k, v);
      };
      globalThis.localStorage = storage;
      const stored = () => JSON.parse(storage.getItem(SHARED_KEY) ?? '[]');
      let requests = await openWithRequests(handler, { restored: false });
      const $ = (id) => doc.getElementById(id);
      const asked = () => requests.filter((r) => r.url === '/api/contributors/photos');
      const status = () => text($('upload-status'));
      const message = () => {
        const node = $('shared-own').querySelector('.form-status');
        return node ? text(node) : null;
      };

      // on load: asked once, in a POST body; the server holds what the device lists: nothing found, nothing rewritten
      assert.equal(asked().length, 1);
      assert.equal(asked()[0].init.method, 'POST');
      assert.deepEqual(JSON.parse(asked()[0].init.body), { contributor_token: TOKEN, landmark_id: 'lm' });
      assert.ok(requests.every((r) => !r.url.includes(TOKEN)), 'the token is in no URL');
      assert.ok(requests.every((r) => !JSON.stringify(r.init.headers ?? {}).includes(TOKEN)), 'nor in a header');
      assert.deepEqual(sharedNames(doc), ['IMG_1.jpg']);
      assert.equal(message(), null);
      assert.deepEqual(stored(), [mine], 'as it was stored');
      assert.equal(writes, 0, 'the browser holds this list already: nothing written');

      // an upload whose answer never arrives: the connection drops after the server stored two photos
      uploadAnswer = () => {
        server = [...server, own('p2'), own('p3')];
        throw new TypeError('Failed to fetch');
      };
      submitUpload(doc);
      await settle();
      assert.equal(asked().length, 2, 'asked at once');
      assert.equal(writes, 1, 'one write for the two photos');
      // (the page asks again for a while, and says the rest can be uploaded again once nothing new came: next test)
      assert.equal(status(), w.storing(2));
      assert.equal($('upload-status').className.trim(), 'form-status is-error');
      assert.equal($('upload-results').hasChildNodes(), false, 'no results to show');
      assert.deepEqual(sharedNames(doc), ['IMG_1.jpg', w.unnamed, w.unnamed]);
      assert.match(text($('shared-own')), w.shared(3));
      assert.equal(message(), w.recovered(2));
      assert.deepEqual(stored(), [
        mine,
        { photo_id: 'p2', token: manage('p2'), filename: '', public_id: own('p2').public_id },
        { photo_id: 'p3', token: manage('p3'), filename: '', public_id: own('p3').public_id },
      ]);
      assert.equal($('shared-own').querySelector('.shared-own-warning'), null, 'the browser keeps this list');
      // its row of the public poses is marked as this device's own, by the public id that came back
      const chips = () => $('pose-rows').querySelectorAll('tr').map((tr) => tr.querySelectorAll('.chip-mine').map(text));
      assert.deepEqual(chips(), [[w.mine], []]);

      // a recovered row is withdrawn with the manage token the server gave again, and leaves the list
      clickWithdraw(doc, 1, w.one);
      await settle();
      const post = requests.filter((r) => r.url === '/api/photos/p2/share_location').at(-1);
      assert.equal(post.init.method, 'POST');
      assert.deepEqual(JSON.parse(post.init.body), { share: false, token: manage('p2') });
      assert.deepEqual(stored().map((s) => s.photo_id), ['p1', 'p3']);
      assert.deepEqual(sharedNames(doc), ['IMG_1.jpg', w.unnamed]);
      assert.deepEqual(chips(), [[], []]);

      // a server error after the photo was stored (the review's second case): the same
      let before = asked().length;
      uploadAnswer = () => {
        server = [...server, own('p4')];
        return json({ detail: 'Internal Server Error' }, 500);
      };
      submitUpload(doc);
      await settle();
      assert.equal(asked().length, before + 1);
      assert.equal(status(), w.storing(1));
      assert.equal(message(), w.recovered(1));
      assert.deepEqual(stored().map((s) => s.photo_id), ['p1', 'p3', 'p4']);
      await pass(RECOVER_PAUSES[0]);
      assert.equal(asked().length, before + 2, 'asked again after the first pause');
      assert.equal(status(), w.lost(1), 'nothing new: the rest can be uploaded again');

      // no answer and nothing stored: the upload failed, and the page says just that
      before = asked().length;
      uploadAnswer = () => {
        throw new TypeError('Failed to fetch');
      };
      submitUpload(doc);
      await settle();
      assert.equal(asked().length, before + 1);
      assert.equal(status(), w.unreachable);
      uploadAnswer = () => json({ detail: 'Internal Server Error' }, 500);
      submitUpload(doc);
      await settle();
      assert.equal(asked().length, before + 2);
      assert.equal(status(), w.server);
      await passRecoverPauses();
      assert.equal(asked().length, before + 2 + RECOVER_PAUSES.length);
      assert.equal(status(), w.server, 'the server stored nothing: the status stays');
      // a refusal (4xx) stored nothing: the server is not asked (no asks of an earlier upload are under way)
      before = asked().length;
      for (const [code, body] of [[413, { detail: 'request body over 1 bytes' }], [404, { detail: "unknown landmark 'lm'" }], [422, { detail: [{ type: 'missing', loc: ['body', 'files'], msg: 'x', input: null }] }]]) {
        uploadAnswer = () => json(body, code);
        submitUpload(doc);
        await settle();
        assert.match(status(), w.tooLarge);
        assert.equal(asked().length, before, String(code));
      }

      // an upload that is answered asks too: what an earlier, unanswered upload stored is listed with it
      uploadAnswer = () => {
        server = [...server, own('p5'), own('p6')]; // p5: stored by an upload whose answer was lost; p6: this one
        return json({ results: [{ ...uploadResult(6, true, manage('p6')), public_id: own('p6').public_id }], triggered_jobs: [] });
      };
      submitUpload(doc);
      await settle();
      assert.equal(asked().length, before + 1);
      assert.deepEqual(sharedNames(doc), ['IMG_1.jpg', w.unnamed, w.unnamed, 'IMG_6.jpg', w.unnamed]);
      assert.deepEqual(stored().map((s) => [s.photo_id, s.filename]), [['p1', 'IMG_1.jpg'], ['p3', ''], ['p4', ''], ['p6', 'IMG_6.jpg'], ['p5', '']]);
      assert.equal(message(), w.recovered(1));

      // what the answer may hold besides: photos of another landmark and rows that are no photos are left out; a
      // listed photo keeps its file name and takes the token and the public id the server gives now
      server = [
        own('p1', { manage_token: 'f'.repeat(64), public_id: 'abcabcabcabc' }),
        own('p7', { landmark_id: 'other' }),
        { photo_id: 'p8', landmark_id: 'lm' },
        { landmark_id: 'lm', manage_token: 'e'.repeat(64) },
        null,
        'p9',
      ];
      before = asked().length;
      doc.visibilityState = 'hidden';
      doc.dispatch('visibilitychange');
      await settle();
      assert.equal(asked().length, before, 'not while hidden');
      doc.visibilityState = 'visible';
      doc.dispatch('visibilitychange');
      await settle();
      assert.equal(asked().length, before + 1, 'when the tab is shown again');
      assert.deepEqual(stored().map((s) => s.photo_id), ['p1', 'p3', 'p4', 'p6', 'p5'], 'only adds: an absent photo may be an earlier registration\'s');
      assert.deepEqual(stored()[0], { photo_id: 'p1', token: 'f'.repeat(64), filename: 'IMG_1.jpg', public_id: 'abcabcabcabc' });
      assert.deepEqual(sharedNames(doc)[0], 'IMG_1.jpg');

      // shown again: also what the browser holds now (a change this tab had no event for)
      storage.setItem(SHARED_KEY, JSON.stringify(stored().slice(0, 2)));
      server = [];
      doc.dispatch('visibilitychange');
      await settle();
      assert.deepEqual(sharedNames(doc), ['IMG_1.jpg', w.unnamed]);

      // an answer made before a withdrawal and arriving after it does not bring the withdrawn photo back
      let release;
      const stale = [own('p1'), own('p3')];
      listAnswer = () =>
        new Promise((resolve) => {
          release = () => resolve(json({ photos: stale }));
        });
      doc.dispatch('visibilitychange'); // asked; the answer is on its way
      await settle();
      listAnswer = null;
      clickWithdraw(doc, 1, w.one); // p3 is withdrawn meanwhile
      await settle();
      assert.deepEqual(stored().map((s) => s.photo_id), ['p1']);
      release();
      await settle();
      assert.deepEqual(stored().map((s) => s.photo_id), ['p1'], 'p3 stays off the list');
      assert.deepEqual(sharedNames(doc), ['IMG_1.jpg']);

      // a registration the server no longer knows, an answer that is no list, no answer: quiet, nothing changes
      const listed = sharedNames(doc);
      for (const answer of [
        () => json({ detail: [{ type: 'register_required', loc: ['body', 'contributor_token'], msg: 'x', input: null }] }, 422),
        () => json({ photos: 'p1' }),
        () => json(null),
        () => {
          throw new TypeError('Failed to fetch');
        },
      ]) {
        listAnswer = answer;
        before = asked().length;
        doc.dispatch('visibilitychange');
        await settle();
        assert.equal(asked().length, before + 1);
        assert.deepEqual(sharedNames(doc), listed);
        assert.equal(cardMode(), 'ready', 'the registration stays: an upload reports one the server does not know');
      }
      listAnswer = null;

      // the form does not wait for the answer about the registration's photos: after an upload without an answer it
      // is usable at once; and an answer that comes after the status moved on changes the list, not the status
      server = [own('p1'), own('p7')];
      uploadAnswer = () => {
        throw new TypeError('Failed to fetch');
      };
      listAnswer = () =>
        new Promise((resolve) => {
          release = () => resolve(json({ photos: server }));
        });
      submitUpload(doc);
      await settle();
      assert.equal(status(), w.unreachable);
      const submit = $('upload-form').querySelector('button');
      assert.equal(submit.disabled, false, 'usable while the answer is on its way');
      assert.equal($('upload-form').hasAttribute('aria-busy'), false);
      listAnswer = null;
      // after forgetting the registration nothing is asked any more; the list stays
      before = asked().length;
      $('forget-registration').click();
      await settle();
      const forgotten = status();
      assert.match(forgotten, lang === 'zh' ? /^已忘记这台设备上的登记。/ : /^This device's registration is forgotten\./);
      release();
      await settle();
      assert.equal(status(), forgotten, 'the status has moved on: left as it is');
      listed.push(w.unnamed); // p7, which the upload stored: withdrawable all the same
      assert.deepEqual(sharedNames(doc), listed);
      doc.dispatch('visibilitychange');
      await settle();
      assert.equal(asked().length, before);
      assert.deepEqual(sharedNames(doc), listed);
      // shown again without a registration: still what the browser holds now (a change this page had no event for)
      storage.setItem(SHARED_KEY, JSON.stringify([...stored(), { photo_id: 'p8', token: manage('p8'), filename: 'IMG_8.jpg', public_id: null }]));
      doc.dispatch('visibilitychange');
      await settle();
      assert.equal(asked().length, before);
      assert.deepEqual(sharedNames(doc), [...listed, 'IMG_8.jpg']);
      if (lang === 'en') assertNoChinese(doc.body, 'recovered photos');

      // a device that lost the whole list (cleared for this landmark, or the answers never came): on load the
      // registration brings every shared photo back
      storage.setItem(CONTRIBUTOR_KEY, JSON.stringify({ token: TOKEN, nickname: 'u', terms: 'terms-v0' }));
      storage.removeItem(SHARED_KEY);
      server = [own('p3'), own('p4')];
      requests = await openWithRequests(handler, { restored: false });
      assert.equal(cardMode(), 'accept', 'an older agreement version: the photos are found all the same');
      assert.deepEqual(sharedNames(doc), [w.unnamed, w.unnamed]);
      assert.equal(text(doc.getElementById('shared-own').querySelector('.form-status')), w.recovered(2));
      assert.deepEqual(stored().map((s) => s.token), [manage('p3'), manage('p4')]);
      // not registered: nothing is asked
      storage.removeItem(CONTRIBUTOR_KEY);
      requests = await openWithRequests(handler, { restored: false });
      assert.equal(requests.filter((r) => r.url === '/api/contributors/photos').length, 0);
      assert.deepEqual(sharedNames(doc), [w.unnamed, w.unnamed], 'the list is this device\'s, registered or not');
    });
  }
});

test('landmark page: after an upload that got no answer the page asks for the registration\'s photos again for a while, because the server may still be storing the batch: the photos are listed as they come, and the rest "can be uploaded again" only once an answer brought nothing new (D34 review, Chinese and English)', async () => {
  const want = {
    zh: {
      unnamed: '照片（没有文件名）',
      storing: (k) => `没有收到上传的结果，但服务器目前已经收下了 ${k} 张照片：它们列在下面，可以撤回。服务器可能还在处理其余的，页面会接着查。`,
      lost: (k) => `没有收到上传的结果，但服务器已经收下了 ${k} 张照片：它们列在下面，可以撤回。还没收下的可以再传一次。`,
      recovered: (k) => `按这台设备的登记，从服务器找回了 ${k} 张照片（没有文件名）。`,
      unreachable: '上传失败：无法连接服务器，请检查网络后重试',
      tooLarge: '上传失败：上传的内容太大，请少选几张照片后重试',
      accepted: '1 张通过。',
      one: '撤回',
    },
    en: {
      unnamed: 'Photo (no file name)',
      storing: (k) =>
        k === 1
          ? "The upload's result did not arrive, but the server has stored 1 photo so far: it is listed below and can be withdrawn. The server may still be working on the rest; this page keeps checking."
          : `The upload's result did not arrive, but the server has stored ${k} photos so far: they are listed below and can be withdrawn. The server may still be working on the rest; this page keeps checking.`,
      lost: (k) =>
        k === 1
          ? "The upload's result did not arrive, but the server did store 1 photo: it is listed below and can be withdrawn. You can upload the rest again."
          : `The upload's result did not arrive, but the server did store ${k} photos: they are listed below and can be withdrawn. You can upload the rest again.`,
      recovered: (k) =>
        k === 1
          ? "Found 1 photo on the server under this device's registration (without its file name)."
          : `Found ${k} photos on the server under this device's registration (without their file names).`,
      unreachable: 'Upload failed: Could not reach the server. Check your connection and try again.',
      tooLarge: 'Upload failed: The upload is too large. Select fewer photos and try again.',
      accepted: '1 photo accepted.',
      one: 'Withdraw',
    },
  };
  const [P0, P1, P2, P3, P4] = RECOVER_PAUSES;
  const TOTAL = P0 + P1 + P2 + P3 + P4;
  const UNTIL_KEY = 'crowd3d.recover_until.lm'; // until when the page asks again, for a page reloaded meanwhile
  const manage = (id) => id.slice(1).repeat(64).slice(0, 64); // the manage token of photo "pN"
  const own = (id) => ({ photo_id: id, landmark_id: 'lm', public_id: `${id}pub`.padEnd(12, '0'), manage_token: manage(id) });
  const upTo = (n) => Array.from({ length: n }, (_, i) => own(`p${i + 1}`)); // the server stores photos p1, p2, … in order
  const cut = () => {
    throw new TypeError('Failed to fetch');
  };
  for (const lang of ['zh', 'en']) {
    await inLang(lang, async () => {
      const w = want[lang];
      let server; // the photos the server holds for this registration now
      let listFails; // asking for them gets no answer
      let listHeld; // asking for them is answered when the test releases it: a function -> Promise of a Response
      let uploadAnswer; // what an upload does: a function -> Response (or throws)
      let requests;
      const handler = (url) => {
        if (url === '/api/landmarks/lm') return json(landmarkDetail({ name: 'Tower' }));
        if (url === '/api/landmarks/lm/poses') return json(posesDoc({ photos: [] }));
        if (url === '/api/contributors/photos' && listHeld) return listHeld();
        if (url === '/api/contributors/photos') return listFails ? cut() : json({ photos: server });
        if (url === '/api/contributors') return json({ nickname: 'v', terms: 'terms-v1', contributor_token: 'c4'.repeat(32) }, 201);
        if (url === '/api/photos') return uploadAnswer();
        const withdrawn = url.match(/^\/api\/photos\/(p\d+)\/share_location$/);
        if (withdrawn) {
          server = server.filter((s) => s.photo_id !== withdrawn[1]);
          return json({ photo_id: withdrawn[1], share_location: false });
        }
        return json({ detail: 'Not Found' }, 404);
      };
      /** A page just opened on a registered device that lists `listed` (and stores `extra`), the server holding `held` for it. */
      const start = async ({ held = [], listed = [], fails = false, extra = {}, registered = true } = {}) => {
        server = held;
        listFails = fails;
        listHeld = null;
        uploadAnswer = cut;
        const entries = { ...(listed.length ? { [SHARED_KEY]: JSON.stringify(listed) } : {}), ...extra };
        globalThis.localStorage = registered ? registeredStorage('u', entries) : fakeStorage(entries);
        requests = await openWithRequests(handler, { restored: false });
      };
      /** The page is loaded again: what the browser stored stays, the page's memory and its timers go. */
      const reload = async () => {
        requests = await openWithRequests(handler, { restored: false });
      };
      const until = () => localStorage.getItem(UNTIL_KEY);
      const $ = (id) => doc.getElementById(id);
      const asks = () => requests.filter((r) => r.url === '/api/contributors/photos');
      const asked = () => asks().length;
      const status = () => text($('upload-status'));
      const names = () => sharedNames(doc);
      const message = () => {
        const node = $('shared-own').querySelector('.form-status');
        return node ? text(node) : null;
      };

      // 1. The connection is reset as soon as the upload is sent; the server stores its eight photos within a second
      // (the review's case). The one ask at the failure finds nothing: the page asks again by itself.
      await start();
      assert.equal(asked(), 1, 'on load');
      assert.equal(until(), null);
      const sent = Date.now();
      submitUpload(doc);
      await settle();
      assert.equal(asked(), 2, 'asked at once');
      assert.equal(status(), w.unreachable);
      assert.deepEqual(names(), []);
      // until when it asks is written down for a reload (7. below): the last pause from now
      assert.match(until(), /^\d+$/);
      assert.ok(Number(until()) >= sent + TOTAL && Number(until()) <= Date.now() + TOTAL);
      server = upTo(8); // stored while nobody asked
      await pass(P0 - 1);
      assert.equal(asked(), 2, 'not before the first pause is over');
      assert.equal(status(), w.unreachable);
      await pass(1);
      assert.equal(asked(), 3);
      assert.deepEqual(names(), Array(8).fill(w.unnamed), 'listed without the user doing anything');
      assert.equal(status(), w.storing(8), 'more may come: nothing yet about uploading the rest again');
      assert.equal($('upload-status').className.trim(), 'form-status is-error');
      assert.equal(message(), w.recovered(8));
      await pass(P1 - 1);
      assert.equal(asked(), 3, 'the second pause is longer');
      await pass(1);
      assert.equal(asked(), 4);
      assert.equal(status(), w.lost(8), 'an answer brought nothing new: the rest can be uploaded again');
      assert.equal($('upload-status').className.trim(), 'form-status is-error');
      for (const [i, ms] of [P2, P3, P4].entries()) {
        await pass(ms - 1);
        assert.equal(asked(), 4 + i, 'each pause longer than the one before');
        await pass(1);
        assert.equal(asked(), 5 + i);
        assert.equal(status(), w.lost(8));
      }
      assert.equal(until(), null, 'over: no longer written down');
      await pass(60 * 60 * 1000);
      assert.equal(asked(), 2 + RECOVER_PAUSES.length, 'a bounded number of asks');
      for (const r of asks()) {
        assert.equal(r.init.method, 'POST');
        assert.deepEqual(JSON.parse(r.init.body), { contributor_token: TOKEN, landmark_id: 'lm' });
      }
      assert.ok(requests.every((r) => !r.url.includes(TOKEN) && !JSON.stringify(r.init.headers ?? {}).includes(TOKEN)), 'the token in no URL or header');
      assert.equal(JSON.parse(localStorage.getItem(SHARED_KEY)).length, 8, 'kept on this device, with their tokens');
      if (lang === 'en') assertNoChinese(doc.body, 'the asks after a lost upload');

      // 2. The connection is reset while the server stores sixteen photos one by one (the review's second
      // measurement): the ask at the failure finds two. Fourteen are still to come: the status must not say that
      // the rest can be uploaded again.
      await start();
      uploadAnswer = () => {
        server = upTo(2);
        return cut();
      };
      submitUpload(doc);
      await settle();
      assert.equal(status(), w.storing(2));
      assert.equal(message(), w.recovered(2));
      server = upTo(5);
      await pass(P0);
      assert.equal(status(), w.storing(5));
      assert.equal(message(), w.recovered(5), 'what these asks found together, not the last one alone');
      server = upTo(16);
      await pass(P1);
      assert.equal(status(), w.storing(16), 'still growing');
      await pass(P2);
      assert.equal(status(), w.lost(16), 'stopped growing');
      assert.equal(names().length, 16);
      // a photo that still comes after that is listed too and the status follows; the last ask settles it
      server = upTo(17);
      await pass(P3);
      assert.equal(status(), w.storing(17));
      server = upTo(18);
      const theirUntil = String(Number(until()) + 5000); // another tab's upload got no answer five seconds later
      localStorage.setItem(UNTIL_KEY, theirUntil);
      await pass(P4);
      assert.equal(status(), w.lost(18), 'the asks are over: this is what the server stored');
      assert.equal(names().length, 18);
      assert.equal(message(), w.recovered(18));
      assert.equal(until(), theirUntil, "the other tab's time is not this page's to drop");

      // 3. An ask that got no answer is no sign that the server stored nothing more (the network is still down).
      // (Here the browser also refuses to keep until when the page asks: it asks all the same.)
      await start();
      const setItem = localStorage.setItem;
      localStorage.setItem = (k, v) => {
        if (k === UNTIL_KEY) throw new DOMException('quota', 'QuotaExceededError');
        return setItem(k, v);
      };
      listFails = true;
      submitUpload(doc);
      await settle();
      assert.equal(asked(), 2);
      assert.equal(status(), w.unreachable);
      server = upTo(3);
      await pass(P0);
      assert.equal(asked(), 3, 'asked, without an answer');
      assert.equal(status(), w.unreachable);
      assert.deepEqual(names(), []);
      listFails = false;
      await pass(P1);
      assert.equal(status(), w.storing(3), 'found once the server could be reached again');
      listFails = true;
      await pass(P2);
      assert.equal(asked(), 5);
      assert.equal(status(), w.storing(3), 'no answer: not "nothing new"');
      await pass(P3);
      assert.equal(status(), w.storing(3));
      await pass(P4);
      assert.equal(asked(), 7);
      assert.equal(status(), w.lost(3), 'the asks are over, answered or not: the page no longer says it keeps checking');

      // 4. What the status counts: the photos the server gave back since this upload failed. Not a row found on an
      // earlier day (listed before, without a name), not a photo another tab uploaded meanwhile (it has its name).
      // It is what the server stored of the upload: withdrawing a photo found does not take it back.
      const earlier = { photo_id: 'p40', token: manage('p40'), filename: '', public_id: own('p40').public_id };
      await start({ held: [own('p40')], listed: [earlier] });
      assert.deepEqual(names(), [w.unnamed]);
      uploadAnswer = () => {
        server = [own('p40'), ...upTo(1)];
        return cut();
      };
      submitUpload(doc);
      await settle();
      assert.deepEqual(names(), [w.unnamed, w.unnamed]);
      assert.equal(status(), w.storing(1), 'one photo of this upload so far');
      assert.equal(message(), w.recovered(1));
      // another tab uploads a photo and stores it with its name; this tab is shown again and reads the list
      const theirs = { photo_id: 'p41', token: manage('p41'), filename: 'IMG_41.jpg', public_id: own('p41').public_id };
      localStorage.setItem(SHARED_KEY, JSON.stringify([...JSON.parse(localStorage.getItem(SHARED_KEY)), theirs]));
      server = [...server, own('p41'), own('p2')];
      listFails = true; // (showing the tab asks too; here only the list the browser holds counts)
      doc.dispatch('visibilitychange');
      await settle();
      listFails = false;
      assert.deepEqual(names(), [w.unnamed, w.unnamed, 'IMG_41.jpg']);
      await pass(P0);
      assert.deepEqual(names(), [w.unnamed, w.unnamed, 'IMG_41.jpg', w.unnamed]);
      assert.equal(status(), w.storing(2), 'p1 and p2: not the row of an earlier day, not the other tab\'s photo');
      assert.equal(message(), w.recovered(2));
      clickWithdraw(doc, 3, w.one); // p2
      await settle();
      clickWithdraw(doc, 1, w.one); // p1
      await settle();
      assert.deepEqual(names(), [w.unnamed, 'IMG_41.jpg']);
      assert.deepEqual(server.map((s) => s.photo_id), ['p40', 'p41']);
      assert.equal(status(), w.storing(2));
      await pass(P1);
      assert.equal(status(), w.lost(2), 'nothing new came; two is still what the server stored of this upload');
      server = [...server, own('p3')];
      await pass(P2);
      assert.equal(status(), w.storing(3), 'a third one: counted with the two withdrawn');
      assert.deepEqual(names(), [w.unnamed, 'IMG_41.jpg', w.unnamed]);
      assert.equal(message(), w.recovered(1), 'under the list: what is listed of it now');

      // 5. A new upload while the page still asks: no ask while it runs (its own photos would be listed without a
      // name, to be named a moment later); as soon as it has ended the asks start again, without touching its status
      await start();
      submitUpload(doc); // the first upload: cut, nothing stored yet
      await settle();
      assert.equal(asked(), 2);
      let release;
      let releaseList;
      listHeld = () =>
        new Promise((resolve) => {
          releaseList = () => resolve(json({ photos: [] }));
        });
      await pass(P0);
      assert.equal(asked(), 3, 'an ask is on its way');
      listHeld = null;
      uploadAnswer = () =>
        new Promise((resolve) => {
          release = () => resolve(json({ results: [{ ...uploadResult(30, true, manage('p30')), public_id: own('p30').public_id }], triggered_jobs: [] }));
        });
      submitUpload(doc); // the second one runs
      await settle();
      releaseList(); // the answer to the ask that was on its way when the second upload started: no further ask from it
      await settle();
      server = upTo(2); // the first upload's photos are being stored meanwhile
      await pass(P0);
      await pass(P1);
      await pass(P2);
      assert.equal(asked(), 3, 'no ask while an upload runs');
      assert.deepEqual(names(), []);
      server = [...upTo(2), own('p30')];
      release();
      await settle();
      assert.equal(asked(), 4, 'asked as soon as it has ended');
      assert.deepEqual(names(), ['IMG_30.jpg', w.unnamed, w.unnamed]);
      assert.equal(status(), w.accepted, "the second upload's own status");
      assert.equal(message(), w.recovered(2));
      server = [...upTo(3), own('p30')]; // and the first upload's third photo after that
      await pass(P0 - 1);
      assert.equal(asked(), 4);
      await pass(1);
      assert.equal(asked(), 5, 'and again after the first pause');
      assert.deepEqual(names(), ['IMG_30.jpg', w.unnamed, w.unnamed, w.unnamed]);
      assert.equal(message(), w.recovered(3));
      assert.equal(status(), w.accepted);
      for (const ms of [P1, P2, P3, P4]) await pass(ms);
      assert.equal(asked(), 4 + RECOVER_PAUSES.length, 'every pause, although nothing new came');
      await pass(60 * 60 * 1000);
      assert.equal(asked(), 4 + RECOVER_PAUSES.length);
      assert.equal(status(), w.accepted);
      // the same after an upload the server refused (it stored nothing of that one, but may store the earlier one's)
      await start();
      submitUpload(doc);
      await settle();
      uploadAnswer = () => json({ detail: 'request body over 1 bytes' }, 413);
      submitUpload(doc);
      await settle();
      assert.equal(asked(), 3, 'the asks start again');
      assert.equal(status(), w.tooLarge);
      server = upTo(1);
      await pass(P0);
      assert.equal(asked(), 4);
      assert.deepEqual(names(), [w.unnamed]);
      assert.equal(message(), w.recovered(1));
      assert.equal(status(), w.tooLarge, "not their upload's status: left as it is");

      // 6. The registration is forgotten and another one made on this device: nothing is asked with the new one for
      // the old one's upload, and the asks are over
      await start();
      uploadAnswer = () => {
        server = upTo(1);
        return cut();
      };
      submitUpload(doc);
      await settle();
      assert.equal(status(), w.storing(1));
      $('forget-registration').click();
      $('nickname').value = 'v';
      $('agree-terms').checked = true;
      $('register-form').dispatch('submit');
      await settle();
      assert.equal(cardMode(), 'ready');
      const registered = status();
      assert.notEqual(registered, w.storing(1));
      server = upTo(2);
      await passRecoverPauses();
      assert.equal(asked(), 2, 'nothing asked with another registration');
      assert.equal(status(), registered);
      assert.deepEqual(names(), [w.unnamed], 'what was found stays listed');

      // 7. The page is reloaded while these asks are under way (a natural thing to do after "upload failed"): its
      // load alone would ask once and find a part of the batch. It goes on asking, until the time the page before
      // wrote down; that time is not moved, so reloading again and again does not keep the asks going.
      await start();
      uploadAnswer = () => {
        server = upTo(2);
        return cut();
      };
      submitUpload(doc);
      await settle();
      assert.equal(status(), w.storing(2));
      const written = String(Number(until()) - 1000); // (a second earlier, to tell it from a time a reload would write)
      localStorage.setItem(UNTIL_KEY, written);
      await reload();
      assert.equal(asked(), 1, 'on load');
      assert.deepEqual(names(), [w.unnamed, w.unnamed], 'what the page before had found and kept');
      assert.equal(status(), '', 'no upload on this page view: no status');
      server = upTo(9);
      await pass(P0 - 1);
      assert.equal(asked(), 1);
      await pass(1);
      assert.equal(asked(), 2, 'asked again, like the page before would have');
      assert.equal(names().length, 9);
      assert.equal(message(), w.recovered(7));
      assert.equal(status(), '');
      assert.equal(until(), written, 'not moved by the reload');
      for (const ms of [P1, P2, P3]) await pass(ms);
      assert.equal(until(), written);
      await pass(P4);
      assert.equal(asked(), 1 + RECOVER_PAUSES.length);
      assert.equal(until(), null, 'over: dropped');
      await pass(60 * 60 * 1000);
      assert.equal(asked(), 1 + RECOVER_PAUSES.length);
      // without such a time a load asks once, as before, also when it finds photos
      await start({ held: upTo(2) });
      assert.deepEqual(names(), [w.unnamed, w.unnamed]);
      assert.equal(message(), w.recovered(2));
      server = upTo(5);
      await passRecoverPauses();
      assert.equal(asked(), 1);
      assert.equal(names().length, 2);
      // a time that is over, one further away than these asks ever last (the clock was set back), or no time at all
      // is not theirs: one ask, and it is dropped
      for (const value of [String(Date.now() - 1), String(Date.now() + TOTAL + 60_000), 'soon', '']) {
        await start({ held: upTo(1), extra: { [UNTIL_KEY]: value } });
        await passRecoverPauses();
        assert.equal(asked(), 1, value);
        assert.equal(until(), null, value);
      }
      // a device that is not registered (any more) has nothing to ask with: the time is dropped
      await start({ registered: false, extra: { [UNTIL_KEY]: String(Date.now() + 5000) } });
      await passRecoverPauses();
      assert.equal(asked(), 0);
      assert.equal(until(), null);
      // the asks of a reloaded page end at the time written down (here the clock is fake too, and moves with the
      // timers): 11 s from the load, so after the asks on load, at 3 s and at 11 s
      mock.timers.reset();
      fakedTimers = false;
      mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_700_000_000_000 });
      await start({ extra: { [UNTIL_KEY]: String(Date.now() + P0 + P1) } });
      assert.equal(asked(), 1);
      server = upTo(1);
      await pass(P0);
      assert.equal(asked(), 2);
      assert.deepEqual(names(), [w.unnamed]);
      await pass(P1);
      assert.equal(asked(), 3, 'the last one: the time is reached');
      assert.equal(until(), null);
      for (const ms of [P2, P3, P4]) await pass(ms);
      assert.equal(asked(), 3);
      mock.timers.reset(); // (the next page opened gets the plain fake timers again)
    });
  }
});

test('the user agreement: one version in records.py, api.js and both language versions of terms.html, which say what is public, what is not and how to withdraw (D32)', async () => {
  // The agreement version accepted photos record (crowd3d/records.py TERMS_CURRENT) stands for exactly the text of
  // web/terms.html: a change of substance in either language needs a new version, and a new version needs this test
  // changed with it.
  const { readFileSync } = await import('node:fs');
  const { TERMS_VERSION } = await import('../api.js');
  const records = readFileSync(new URL('../../crowd3d/records.py', import.meta.url), 'utf8');
  const current = records.match(/^TERMS_CURRENT = (TERMS_V\d+)$/m);
  assert.ok(current, 'records.py names the current agreement version');
  const value = records.match(new RegExp(`^${current[1]} = "([^"]+)"$`, 'm'));
  assert.equal(value?.[1], 'terms-v1');
  assert.equal(TERMS_VERSION, value[1]);
  const terms = readFileSync(new URL('../terms.html', import.meta.url), 'utf8');
  const articles = [...terms.matchAll(/<article\b([^>]*)>([\s\S]*?)<\/article>/g)].map((m) => ({ attrs: m[1], body: m[2] }));
  assert.equal(articles.length, 2);
  const [zh, en] = articles;
  assert.match(zh.attrs, /data-lang-only="zh"/);
  assert.match(zh.attrs, /lang="zh-CN"/);
  assert.match(en.attrs, /data-lang-only="en"/);
  assert.match(en.attrs, /lang="en"/);
  for (const a of articles) assert.match(a.attrs, new RegExp(`data-terms-version="${value[1]}"`));
  const sections = (a) => [...a.body.matchAll(/<section data-section="([a-z-]+)"/g)].map((m) => m[1]);
  const ids = ['scope', 'public-photo', 'public-pose', 'nickname', 'never-shown', 'withdraw', 'risks', 'licence', 'minors', 'changes', 'contact'];
  assert.deepEqual(sections(zh), ids);
  assert.deepEqual(sections(en), ids);
  const plain = (a) => a.body.replace(/<[^>]+>/g, '').replace(/&nbsp;/g, ' ').replace(/\s+/g, ' ');
  for (const part of ['草案，待法律审核', '1 米', '1°', '水平视角，取整到 1°', '1600', 'EXIF', '撤回', '可辨认的人',
    // section 2 as the code does it (D33): only photos whose position and heading are determined are shown
    '一旦能确定拍摄位置和朝向（见第 3 节），就会公开显示', '确定不了的照片不显示',
    // never shown
    '哪张照片是谁上传的', '上传时间和拍摄时间', '内部的编号', '原始文件', '每个人上传了多少张', '没有通过检查的照片',
    // the remaining linking risk of the upload hint (tests/test_api.py
    // test_anyone_can_move_the_poses_snapshot_soon_after_a_new_upload_as_the_agreement_says)
    '任何人都能用手动重建或者上传几张照片让下一次建模很快开始']) {
    assert.ok(plain(zh).includes(part), part);
  }
  for (const part of ['Draft - pending legal review', '1 m', '1°', 'horizontal field of view, rounded to 1°', '1600 px', 'EXIF', 'withdraw', 'Withdraw all',
    'is shown publicly as soon as its position and heading can be worked out (section 3)', 'Until then it is not shown',
    'who uploaded which photo', 'when a photo was uploaded or taken', 'internal photo IDs', 'the original file',
    'how many photos each person uploaded', 'anyone can make that build start soon']) {
    assert.ok(plain(en).includes(part), part);
  }
  assert.doesNotMatch(plain(en), /[\u4e00-\u9fff]/, 'the English version holds no Chinese');
  // fields of view to whole degrees (D33), no longer 5-degree steps
  for (const a of articles) assert.doesNotMatch(plain(a), /取整到 5°|rounded to 5°/);
  // section 2 no longer says every accepted photo is shown (D33): a photo without a determined pose never is
  assert.doesNotMatch(plain(zh), /每张照片都会公开显示/);
  assert.doesNotMatch(plain(en), /Every photo you upload that passes our checks is shown publicly/);
  // accepted once, when registering (D33): sections 1, 4 and 10 say so, and no section asks for it with every upload
  const section = (a, id) => plain({ body: a.body.match(new RegExp(`<section data-section="${id}">([\\s\\S]*?)</section>`))[1] });
  for (const id of ['scope', 'nickname', 'changes']) {
    assert.match(section(zh, id), /登记/, `zh ${id}`);
    assert.match(section(en, id), /register/, `en ${id}`);
  }
  assert.match(section(zh, 'nickname'), /^ ?4\. 登记和昵称 /);
  assert.match(section(en, 'nickname'), /^ ?4\. Registration and nickname /);
  for (const [id, part] of [
    ['scope', '登记时必须阅读并同意本协议，只需同意一次；不同意就不能登记，也就不能上传照片'],
    // only an accepted photo records the version, until it is withdrawn (a rejected or withdrawn photo keeps none)
    ['scope', '我们记录你同意的协议版本和时间；每张通过检查的照片也记录它通过时依据的协议版本，直到你撤回这张照片。'],
    ['nickname', '登记本身不会公开任何内容'],
    ['nickname', '你在某个地标有照片通过检查后，你的昵称才会出现在那个地标的“共同完成者”名单里'],
    ['nickname', '这台设备的浏览器里会保存一个登记凭证'],
    ['nickname', '服务器只保存它的指纹（哈希），不会再显示它'],
    ['nickname', '清除了浏览器数据、换了设备，或者点了“忘记这台设备上的登记”，就要重新登记、再次同意本协议'],
    ['never-shown', '你的登记信息：登记凭证、登记时间和同意协议的时间'],
    ['withdraw', '忘记这台设备上的登记不会删掉撤回凭证'],
    ['changes', '已经登记的参与者，下次上传前要阅读并同意新版本一次，不用重新登记'],
  ]) {
    assert.ok(section(zh, id).includes(part), part);
  }
  for (const [id, part] of [
    ['scope', "You accept them once, when you register; if you don't, you cannot register and so cannot upload."],
    ['scope', 'We record which version you accepted and when, and each photo that passes our checks records the version it was accepted under, until you withdraw it.'],
    ['nickname', 'Registering publishes nothing'],
    ['nickname', 'only once one of your photos there passes our checks'],
    ['nickname', "this device's browser keeps a registration key"],
    ['nickname', 'our server keeps only a fingerprint of it (a hash) and never shows it again'],
    ['nickname', "If you clear your browser data, switch devices or use Forget this device's registration, you register and accept these terms again."],
    ['never-shown', 'your registration details: your registration key, and when you registered and accepted these terms'],
    ['withdraw', "Forgetting this device's registration does not delete your withdrawal tokens."],
    ['changes', 'If you are already registered, you need to read and accept the new version once before your next upload; you do not need to register again.'],
  ]) {
    assert.ok(section(en, id).includes(part), part);
  }
  assert.doesNotMatch(plain(zh), /下次上传前必须重新阅读并同意|每次上传都会记录你同意的协议版本|上传前必须阅读并同意本协议|每张照片也记录/);
  assert.doesNotMatch(plain(en), /accept it before your next upload|You need to read and accept them before uploading|Each upload records which version|each photo records the version/i);
  // the page's description: accepted when registering
  assert.match(terms, /<meta name="description" content="登记参与者时需要同意的用户协议：/);
  for (const a of articles) assert.match(a.body, /<mark>/, 'placeholders are marked');
});

test('landmark page: the registration form holds the nickname and the agreement box, which links the agreement, is not ticked in the markup, and its hint names what is public, the rounding, the remaining risk, registering once and withdrawal (D32, D33)', async () => {
  const { readFileSync } = await import('node:fs');
  const html = readFileSync(new URL('../landmark.html', import.meta.url), 'utf8');
  const form = (id) => html.match(new RegExp(`<form id="${id}"([^>]*)>([\\s\\S]*?)</form>`));
  const register = form('register-form');
  const upload = form('upload-form');
  assert.ok(register && upload, 'the two forms');
  // only the registration form shows before landmark.js knows this device's registration
  assert.doesNotMatch(register[1], /\bhidden\b/);
  assert.match(upload[1], /\bhidden\b/);
  assert.match(html, /<div id="registered" class="registered" hidden>\s*<span id="upload-as" class="registered-name"><\/span>\s*<button id="forget-registration" type="button" class="btn btn-ghost btn-small" data-i18n="register\.forget">忘记这台设备上的登记<\/button>\s*<\/div>/);
  assert.ok(html.indexOf('<div id="registered"') < html.indexOf('<form id="register-form"'), 'who is registered, above the agreement to accept');
  assert.match(register[2], /<p id="register-intro" class="register-intro" data-i18n="register\.intro">/);
  // the nickname: no required attribute (the page shows its own message; a hidden required field would block accepting)
  const nickname = register[2].match(/<div class="field" id="nickname-field">[\s\S]*?<input id="nickname"([^>]*)>/);
  assert.ok(nickname, 'the nickname field');
  assert.match(nickname[1], /name="nickname" type="text" maxlength="64" autocomplete="nickname"/);
  assert.doesNotMatch(nickname[1], /\brequired\b/);
  assert.match(register[2], /<button id="register-submit" type="submit" class="btn btn-primary" data-i18n="register\.submit">登记<\/button>/);
  // the upload form: the photos only (the nickname and the agreement come from the registration)
  assert.doesNotMatch(upload[2], /agree-terms|nickname|user-id|user_id|terms/);
  assert.match(upload[2], /<input id="files" name="files" type="file"/);
  assert.doesNotMatch(html, /id="user-id"|name="user_id"/);
  const label = register[2].match(/<label class="check"><input id="agree-terms"[^>]*>[\s\S]*?<\/label>/);
  assert.ok(label, 'the agreement checkbox and its label');
  assert.match(label[0], /type="checkbox"/);
  assert.doesNotMatch(label[0], /\bchecked\b/, 'not ticked in the markup');
  assert.doesNotMatch(label[0], /\brequired\b/, 'the page shows its own message');
  assert.match(
    label[0],
    /<span><span data-i18n="register\.terms\.pre">我已阅读并同意<\/span><a href="terms\.html" target="_blank" rel="noopener" data-i18n="register\.terms\.link">《用户协议》<\/a><span data-i18n="register\.terms\.post"><\/span><\/span>/,
  );
  assert.doesNotMatch(html, /share-location|share_location/);
  const hint = register[2].match(/<p id="agree-terms-hint"[^>]*data-i18n="register\.terms\.hint">([^<]*)<\/p>/);
  assert.ok(hint, 'the agreement hint');
  const zh = hint[1].replace(/&#160;/g, '\u00a0'); // a number and its unit: a no-break space
  for (const part of [
    /^上传的照片会公开显示/,
    /去掉 EXIF 等拍摄信息/,
    /长边不超过 1600\u00a0像素/,
    /拍摄位置（取整到 1\u00a0米）/,
    /朝向（取整到 1°）/,
    /视角（取整到 1°）/,
    /视角（取整到 1°）；能确定位置和朝向的照片才显示。你的昵称/, // as the agreement's section 2 (D33)
    /你的昵称会出现在你有照片通过检查的地标的共同完成者里/, // contributors lists the nicknames of accepted photos (D33)
    /不显示哪张照片是谁传的、什么时候传的/,
    /公开的照片和位置要等那个地标下一次建模结束后才显示/, // the public poses are a snapshot (D30, owner question 2)
    // the remaining risk (owner question 2): anyone can end the snapshot soon (tests/test_api.py
    // test_anyone_can_move_the_poses_snapshot_soon_after_a_new_upload_as_the_agreement_says)
    /任何人都能用手动重建或者上传几张照片让下一次建模很快开始/,
    /别人可能把那时新出现的照片和你的昵称对上，猜出哪些是你拍的/,
    /那里上传的人少时更容易/,
    /不同意就不能登记，也就不能上传。/, // accepted at registration (D33)
    /只需登记一次/,
    /上传后可以在这台设备上撤回照片。$/,
  ]) {
    assert.match(zh, part);
  }
  assert.doesNotMatch(zh, /不同意就不能上传；|取整到 5°/);
  for (const page of ['index.html', 'landmark.html']) {
    const markup = readFileSync(new URL(`../${page}`, import.meta.url), 'utf8');
    assert.match(markup, /<footer class="site-footer">[\s\S]*<a href="terms\.html" data-i18n="common\.terms">用户协议<\/a>/, page);
  }
});

test('landmark page: contributors are nicknames by name without counts, even from a server that still sends counts (D30)', async () => {
  globalThis.localStorage = registeredStorage('zoe'); // "you": the nickname this device registered (D33)
  await openLandmark({
    '/api/landmarks/lm': landmarkDetail({ contributors: [{ user_id: 'zoe', photos: 30 }, { user_id: 'amy', photos: 4 }] }),
  });
  const list = doc.getElementById('contributors');
  assert.deepEqual(
    list.querySelectorAll('li').map((li) => li.textContent),
    ['amy', 'zoe（你）'],
  );
  assert.doesNotMatch(text(list), /\d|张/);
  assert.equal(list.querySelectorAll('.contributor.is-me').length, 1);
});

function posesDoc(over = {}) {
  return {
    format: 'crowd3d.photo-poses',
    version: 2,
    landmark: { id: 'lm', name: 'Tower' },
    frame: { name: 'landmark-enu', origin: { lat: 31.2, lon: 121.5 }, axes: 'x east, y north, z up', unit: 'm' },
    as_of: 1_000_000,
    rounding: { position_m: 1, angle_deg: 1, fov_deg: 1, aspect_decimals: 2 },
    uncertainty: {
      sigmas: 2,
      measured_coverage: {
        sfm: { within: 146, n: 165, runs: 2, landmarks: 4, lowest: { within: 1, n: 10, coverage: 0.1 },
               highest: { within: 13, n: 13, coverage: 1 }, rule_calibrated_here: false, coverage: 0.885 },
        loc: null,
        exif: { within: 23, n: 28, runs: 2, landmarks: 3, lowest: { within: 1, n: 2, coverage: 0.5 },
                highest: { within: 1, n: 1, coverage: 1 }, rule_calibrated_here: true, coverage: 0.821 },
      },
      basis: 'commons-ten-v1 test split',
    },
    counts: { photos: 9, shared: 4, shown: 3, sfm: 2, loc: 0, exif: 1, not_determined: 1 },
    photos: [
      { id: '0a0a0a0a0a0a', src: 'sfm', p: [12, -3, 2], yaw: 183, pitch: 5, roll: 0, hfov: 65, aspect: 1.33, sp: 16, sy: 21 },
      { id: '1b1b1b1b1b1b', src: 'exif', p: [-40, 7, null], yaw: 92, pitch: null, roll: null, hfov: null, aspect: 0.75, sp: 41, sy: 31 },
      { id: '2c2c2c2c2c2c', src: 'sfm', p: [0, 25, 1], yaw: 359, pitch: -2, roll: 1, hfov: 60, aspect: 1.33, sp: 16, sy: 21 },
      // a row this page cannot place (an unknown source, no heading) is dropped, never guessed
      { id: '3d3d3d3d3d3d', src: 'gps', p: [1, 2, null], yaw: null, sp: 61, sy: null },
    ],
    ...over,
  };
}

test('landmark page: the public poses table lists source, position, heading and range, marks your own, and no uploader or time (D30)', async () => {
  globalThis.localStorage = fakeStorage({
    [SHARED_KEY]: JSON.stringify([{ photo_id: 'p9', token: 'c'.repeat(64), filename: 'mine.jpg', public_id: '2c2c2c2c2c2c' }]),
  });
  const calls = await openLandmark({ '/api/landmarks/lm': landmarkDetail(), '/api/landmarks/lm/poses': posesDoc() });
  assert.deepEqual(
    calls.map((c) => c.url),
    ['/api/landmarks/lm', '/api/landmarks/lm/poses'],
  );
  assert.equal(doc.getElementById('poses-card').hidden, false);
  assert.equal(
    text(doc.getElementById('poses-count')),
    '公开的 4 张 / 共 9 张，显示 3 张；1 张无法可靠确定位置和朝向，不显示。',
  );
  assert.match(text(doc.getElementById('poses-as-of')), /更新$/);
  const rows = doc.getElementById('pose-rows').querySelectorAll('tr').map((tr) => tr.querySelectorAll('td').map(text));
  assert.deepEqual(rows, [
    ['模型重建', '东 12 米，南 3 米', '183°（南）', '±16 米 · ±21°'],
    ['手机 GPS + 罗盘（未经模型核对）', '西 40 米，北 7 米', '92°（东）', '±41 米 · ±31°'],
    ['模型重建你的', '东 0 米，北 25 米', '359°（北）', '±16 米 · ±21°'],
  ]);
  assert.equal(doc.getElementById('pose-table').hidden, false);
  assert.equal(doc.getElementById('show-poses-label').hidden, false);
  assert.equal(doc.getElementById('poses-snap-note').hidden, true);
  const coverage = text(doc.getElementById('poses-coverage'));
  assert.match(coverage, /^±范围是 2σ。/);
  assert.match(coverage, /模型重建：4 个测试地标、2 次运行合计 165 个中 146 个（88%），单个地标单次运行最低 10 个中 1 个（10%）/);
  assert.match(coverage, /手机 GPS \+ 罗盘（未经模型核对）：3 个测试地标、2 次运行合计 28 个中 23 个（82%）.*不是独立检验/);
  assert.doesNotMatch(coverage, /与模型比对/, 'only the sources shown here');
  assert.match(text(doc.getElementById('poses-sources')), /手机 GPS \+ 罗盘（未经模型核对）：照片自带的定位和指南针/);
  const card = text(doc.getElementById('poses-card'));
  for (const s of ['3d3d3d3d3d3d', '0a0a0a0a0a0a', 'p9', 'mine.jpg']) assert.ok(!card.includes(s), s);
});

test('landmark page: on a narrow card the poses table stacks its rows, each value after its column name, with explicit table roles (D33)', async () => {
  const { readFileSync } = await import('node:fs');
  // style.css: a container query on the poses card, not the window (the main column at 700-1024 px is narrow too); the
  // card's content box is the table box's width
  const css = readFileSync(new URL('../style.css', import.meta.url), 'utf8');
  assert.match(css, /#poses-card \{\s*container-type: inline-size;\s*\}/);
  const block = css.match(/@container \(max-width: 639px\) \{\n([\s\S]*?)\n\}\n/);
  assert.ok(block, 'the stacked layout below 640 px of card width');
  assert.match(block[1], /\.pose-table thead \{\s*position: absolute;[^}]*clip-path: inset\(50%\);/, 'headers visually hidden, still read');
  assert.match(block[1], /\.pose-table,\s*\.pose-table tbody \{\s*display: block;\s*\}/);
  assert.match(block[1], /\.pose-table tr \{\s*display: grid;\s*grid-template-columns: auto minmax\(0, 1fr\);/);
  assert.match(block[1], /\.pose-table td:nth-child\(-n \+ 2\) \{\s*grid-column: 1 \/ -1;\s*\}/);
  assert.match(block[1], /\.pose-table td\[data-label\] \{\s*display: flex;\s*flex-wrap: wrap;/);
  assert.match(block[1], /\.pose-table td\[data-label\]::before \{\s*content: attr\(data-label\);\s*content: attr\(data-label\) \/ "";\s*color: var\(--muted\);\s*\}/);
  // only theme tokens: light and dark both work
  assert.doesNotMatch(block[1], /#[0-9a-f]{3,8}\b|rgb\(|hsl\(/i);
  // the scroll box shows a few of many rows at every width, stacked or a table (D33, both screenshot rounds): it has
  // a frame, and its last child, a fade that sticks to its bottom edge, marks the row cut there; outside the
  // container query, so the table layout of a wide card has them too. The fade overlaps only the last row's bottom
  // padding (8 px), so at the end of the list it covers no text. Nothing of them is left inside the query. The box
  // is 376 px high in the table layout (also a browser without container queries) and 360 px for stacked rows.
  const rule = (selector) => css.match(new RegExp(`\\n${selector.replace(/[.:()]/g, '\\$&')} \\{\\n([^}]*)\\}`))?.[1];
  const wrap = rule('.pose-table-wrap');
  assert.ok(wrap, 'the scroll box, at every width');
  assert.match(wrap, /max-height: 376px;\s*overflow-y: auto;\s*border: 1px solid var\(--border\);\s*border-radius: 8px;\s*padding: 0 12px;/);
  const fade = rule('.pose-table-wrap::after');
  assert.ok(fade, 'the fade, at every width');
  assert.match(fade, /content: "";\s*display: block;\s*position: sticky;\s*bottom: 0;\s*height: 32px;\s*margin-top: -8px;\s*background: linear-gradient\(to bottom, transparent, var\(--surface\)\);\s*pointer-events: none;/);
  assert.match(css, /\.table th,\s*\.table td \{[^}]*padding: 8px 10px 8px 0;/, 'the 8 px under a row\'s text that the fade overlaps');
  // stacked, a cell has no padding: the 8 px is the row's own (and no rule of the query takes it from the last row)
  assert.match(block[1], /\n {2}\.pose-table td \{[^}]*\bpadding: 0;/);
  assert.match(block[1], /\n {2}\.pose-table tr \{[^}]*\n {4}padding: 8px 0;\n[^}]*\}/, 'the 8 px under a stacked row\'s text that the fade overlaps');
  assert.doesNotMatch(block[1], /:last-child \{[^}]*padding/, 'the last stacked row keeps its 8 px');
  for (const r of [wrap, fade]) assert.doesNotMatch(r, /#[0-9a-f]{3,8}\b|rgb\(|hsl\(/i);
  assert.doesNotMatch(block[1], /::after|border: 1px/, 'no frame or fade only for narrow cards');
  // stacked rows keep the D30 360 px: the query sets the box's height, nothing else of it
  assert.deepEqual([...block[1].matchAll(/pose-table-wrap[^{]*\{([^}]*)\}/g)].map((m) => m[1].trim()), ['max-height: 360px;']);
  // in the table layout the frame's padding is the last column's right margin: the widest table fits a 640 px card
  assert.match(css, /\.pose-table th:last-child,\s*\.pose-table td:last-child \{\s*padding-right: 0;\s*\}/);
  // the table layout's cue (D33 final review r3): at the top of a long list the cut row's text must show through
  // the fade. With the D30 360 px the box's bottom fell between the 8th and 9th rows' text, which lay under the
  // fade's opaque end: the top of the list looked like its end. A row (measured in Chromium): its two paddings, its
  // line (the table's font size at the body's line height) or a taller "Yours" chip, and a 1 px rule; the header
  // half a rule less. Whatever the number of chips above it (0-8), some row's line has at least 10 px in the part
  // of the fade that leaves it visible (the fade's top 7/8), inside the 1 px frame.
  const px = (re, what) => { const m = css.match(re); assert.ok(m, what); return Number(m[1]); };
  const fontSize = px(/\n\.table \{[^}]*font-size: ([\d.]+)px;/, 'the table font size');
  const line = fontSize * px(/\nbody \{[^}]*line-height: ([\d.]+);/, 'the body line height');
  const pad = px(/\.table th,\s*\.table td \{[^}]*padding: ([\d.]+)px /, 'the cell padding');
  const chip = px(/\n\.chip \{[^}]*line-height: ([\d.]+)px;/, 'the chip line') + 2 * px(/\n\.chip \{[^}]*padding: ([\d.]+)px /, 'the chip padding');
  const boxInner = Number(wrap.match(/max-height: ([\d.]+)px;/)[1]) - 2;
  const fadeH = Number(fade.match(/\n {2}height: ([\d.]+)px;/)[1]);
  const row = 2 * pad + line + 1;
  const visible = (lineTop) => Math.min(lineTop + line, boxInner - fadeH / 8) - Math.max(lineTop, boxInner - fadeH);
  for (let chips = 0; chips <= 8; chips += 1) {
    let best = -Infinity;
    for (let i = 0; i < 20; i += 1) {
      const top = row - 0.5 + i * row + Math.min(i, chips) * Math.max(0, chip - line);
      best = Math.max(best, visible(top + pad));
    }
    assert.ok(best >= 10, `the cut row's text shows through the fade with ${chips} chips above it (${best.toFixed(1)} px)`);
  }

  // landmark.html: the roles a table keeps whatever its display
  const html = readFileSync(new URL('../landmark.html', import.meta.url), 'utf8');
  const table = html.match(/<table class="table pose-table"[^>]*>[\s\S]*?<\/table>/)?.[0];
  assert.ok(table, 'the poses table');
  assert.match(table, /^<table class="table pose-table" role="table"/);
  assert.match(table, /<thead role="rowgroup"><tr role="row">/);
  assert.match(table, /<tbody id="pose-rows" role="rowgroup"><\/tbody>/);
  assert.equal(table.match(/<th\b/g).length, 4);
  assert.equal(table.match(/<th scope="col" role="columnheader"/g).length, 4);

  // landmark.js renderPoses: the three value cells carry their column's name, the source cell none
  const want = {
    zh: ['位置（离中心）', '朝向', '±范围'],
    en: ['Position (from the centre)', 'Heading', '± range'],
  };
  for (const l of ['zh', 'en']) {
    await inLang(l, async () => {
      globalThis.localStorage = fakeStorage();
      await openLandmark({ '/api/landmarks/lm': landmarkDetail(), '/api/landmarks/lm/poses': posesDoc() });
      assert.deepEqual(want[l], [t('poses.col.position'), t('poses.col.heading'), t('poses.col.range')]);
      const rows = doc.getElementById('pose-rows').querySelectorAll('tr');
      assert.equal(rows.length, 3);
      for (const tr of rows) {
        assert.equal(tr.getAttribute('role'), 'row');
        const cells = tr.querySelectorAll('td');
        assert.deepEqual(cells.map((c) => c.getAttribute('role')), ['cell', 'cell', 'cell', 'cell']);
        assert.deepEqual(cells.map((c) => c.getAttribute('data-label')), [null, ...want[l]], l);
      }
    });
    doc.restore();
  }
});

test('landmark page: no pose markers on a snapped model\'s point cloud; the page says why (D30)', async () => {
  const snapped = { status: 'ready', url: '/files/models/lm/m0.glb', registration: { accepted: true, reason: null },
    poses_url: '/api/landmarks/lm/poses', snapped: true };
  await openLandmark({
    '/api/landmarks/lm': landmarkDetail({ models: [{ ...model('m0', 'face', 0), stylized: snapped }, model('m1', 'face', 1)] }),
    '/api/landmarks/lm/poses': posesDoc(),
  });
  assert.equal(doc.getElementById('pose-table').hidden, false); // the table is in landmark ENU: it stays
  assert.equal(doc.getElementById('show-poses-label').hidden, true);
  const note = doc.getElementById('poses-snap-note');
  assert.equal(note.hidden, false);
  assert.match(text(note), /按地图轮廓对齐过.*点云没有.*风格化模型里看/);
});

test('landmark page: before any build finished, the poses card says when shared photos appear (D30)', async () => {
  const empty = posesDoc({ as_of: null, photos: [], counts: { photos: 0, shared: 0, shown: 0, sfm: 0, loc: 0, exif: 0, not_determined: 0 } });
  await openLandmark({ '/api/landmarks/lm': landmarkDetail(), '/api/landmarks/lm/poses': empty });
  assert.equal(doc.getElementById('poses-card').hidden, false);
  assert.equal(text(doc.getElementById('poses-count')), '这里还没有建模完成过；公开的照片会在第一次建模结束后显示。');
  assert.equal(doc.getElementById('pose-table').hidden, true);
  assert.equal(doc.getElementById('show-poses-label').hidden, true);
  assert.equal(text(doc.getElementById('poses-coverage')), '');
});

test('landmark page: a poses document of another format or version is not drawn (D30)', async () => {
  await openLandmark({ '/api/landmarks/lm': landmarkDetail(), '/api/landmarks/lm/poses': posesDoc({ version: 1 }) });
  assert.equal(doc.getElementById('poses-card').hidden, true);
});

test('landmark page: the public poses are fetched with full refreshes only, not with job polls (D30)', async () => {
  mock.timers.enable({ apis: ['setTimeout', 'Date'], now: 1_000_000 });
  const running = job({ id: 'jr', kind: 'face', sector: 0, status: 'running', finished_at: null });
  const calls = await openLandmark({
    '/api/landmarks/lm': landmarkDetail({ jobs: [running] }),
    '/api/landmarks/lm/poses': posesDoc(),
    '/api/jobs/jr': running,
  });
  mock.timers.tick(3000);
  await settle();
  mock.timers.tick(3000);
  await settle();
  const urls = calls.map((c) => c.url);
  assert.ok(urls.filter((u) => u === '/api/jobs/jr').length >= 2, urls.join());
  assert.equal(urls.filter((u) => u === '/api/landmarks/lm/poses').length, 1);
});

// ---------- the pages in English (D32) ----------

// Chinese characters and CJK / full-width punctuation (as in i18n.test.mjs)
const CJK = /[\p{Script=Han}　-〿＀-￯]/u;

/** Everything a node shows or says: its texts, the markup it was given (SVG) and its attribute values. */
function shown(root) {
  const out = [];
  const walk = (node) => {
    if (node.nodeType !== 1) {
      out.push(node.textContent);
      return;
    }
    for (const v of node.attributes.values()) out.push(v);
    node.childNodes.forEach(walk);
  };
  walk(root);
  return out.filter((s) => s.trim());
}

function assertNoChinese(root, where) {
  const found = shown(root).filter((s) => CJK.test(s));
  assert.deepEqual(found, [], `${where}: no Chinese in the English page`);
}

// Intl writes a narrow no-break space before AM/PM in newer ICU versions
const plainSpaces = (s) => s.replace(/[  ]/g, ' ');

test('home page in English: the list, the map and the create form, without a Chinese character (D32)', async () => {
  await inLang('en', async () => {
    doc = installDom();
    const created = [];
    globalThis.fetch = async (url, init = {}) => {
      if (url === '/api/landmarks' && init.method === 'POST') {
        created.push(JSON.parse(init.body));
        const status = created.length === 1 ? 422 : created.length === 2 ? 409 : 200;
        const body = status === 422
          ? { detail: [{ type: 'less_than_equal', loc: ['body', 'lat'], msg: 'x', ctx: { le: 90 } }] }
          : status === 409 ? { detail: "landmark 'tower' exists" } : { id: 'tower', name: 'Tower' };
        return new Response(JSON.stringify(body), { status });
      }
      return new Response(JSON.stringify({ landmarks: [
        summary({ id: 'a', name: 'Tower', num_photos: 1, num_contributors: 1, full_ready: false, sectors: SECTORS.map((x) => ({ ...x, has_face_model: false })) }),
        summary({ id: 'b', name: 'Gate', num_photos: 2, num_contributors: 2, full_ready: false }),
      ] }), { status: 200 });
    };
    await import(`../index.js?page=${++page}`);
    await settle();
    const cards = doc.getElementById('landmark-list').querySelectorAll('.lm-card-meta').map((m) => m.querySelectorAll('span').map(text));
    assert.deepEqual(cards, [['1 photo', '1 contributor', 'Collecting photos'], ['2 photos', '2 contributors', '2 facades built']]);
    assert.match(doc.getElementById('landmark-list').querySelector('.progress').getAttribute('aria-label'), /^Coverage of Tower$/);
    assert.equal(text(doc.getElementById('list-count')), '2 landmarks');
    assert.equal(text(doc.getElementById('map')), 'Loading the map…');
    const leaflet = fakeLeaflet();
    globalThis.L = leaflet.L;
    doc.head.querySelector('link').onload();
    doc.head.querySelector('script').onload();
    await settle();
    // Leaflet's own zoom control is off; the page's says it in the page language
    assert.deepEqual(leaflet.zooms, [{ map: { scrollWheelZoom: false, zoomControl: false } }, { zoomInTitle: 'Zoom in', zoomOutTitle: 'Zoom out' }]);
    // the popups fit a phone's map (324 px wide at 390) and clear the zoom buttons: 208 + 48 px of Leaflet's margins
    // = 256 = 324 - 56 - 12
    assert.deepEqual(leaflet.popups, [1, 2].map(() => ({ maxWidth: 208, autoPanPaddingTopLeft: [56, 12], autoPanPaddingBottomRight: [12, 12] })));
    // the close button in the page language; a popup bound on a wider map is fitted again as it opens
    const closeButton = new FakeElement('a');
    closeButton.setAttribute('class', 'leaflet-popup-close-button');
    closeButton.setAttribute('aria-label', 'Close popup');
    const popupEl = new FakeElement('div');
    popupEl.append(closeButton);
    let updates = 0;
    const popup = { options: { maxWidth: 300 }, getElement: () => popupEl, update: () => updates++ };
    for (const fn of leaflet.handlers.popupopen ?? []) fn({ popup });
    assert.equal(closeButton.getAttribute('aria-label'), 'Close');
    assert.equal(closeButton.getAttribute('title'), 'Close');
    assert.deepEqual([popup.options.maxWidth, updates], [208, 1]);
    for (const fn of leaflet.handlers.popupopen) fn({ popup });
    assert.equal(updates, 1, 'no second layout when it fits');

    // the create form: a 422, a 409, then created
    const form = doc.getElementById('new-landmark');
    const button = new FakeElement('button');
    button.setAttribute('type', 'submit');
    form.append(button);
    form.reset = () => {};
    const realFormData = globalThis.FormData;
    globalThis.FormData = class {
      get(k) {
        return { name: ' Tower ', lat: '91', lon: '11.25', radius_m: '150' }[k];
      }
    };
    try {
      const status = () => text(doc.getElementById('new-status'));
      form.dispatch('submit');
      await settle();
      assert.equal(status(), 'Please check the form: Latitude: must be at most 90');
      form.dispatch('submit');
      await settle();
      assert.equal(status(), 'A landmark with the same ID already exists. Choose another name.');
      form.dispatch('submit');
      await settle();
      assert.equal(status(), 'Created "Tower". Upload photos →');
    } finally {
      globalThis.FormData = realFormData;
    }
    assertNoChinese(doc.body, 'home page');
  });
});

test('landmark page in English: the details, registering, upload results, a failed job, your photos, the public poses and the compass, without a Chinese character (D32, D33)', async () => {
  const oct21 = new Date(2026, 9, 21, 9, 5).getTime() / 1000; // local time, like the page
  const now = oct21 + 4 * 86400;
  mock.timers.enable({ apis: ['setTimeout', 'Date'], now: now * 1000 });
  const storage = registeredStorage('u', {
    [SHARED_KEY]: JSON.stringify([{ photo_id: 'p8', token: 'd'.repeat(64), filename: 'mine.jpg', public_id: '2c2c2c2c2c2c' }]),
  });
  globalThis.localStorage = storage;
  let refusal = null; // the type an upload is refused with (D33)
  let registerFails = false;
  const handler = (url, init) => {
    if (url === '/api/landmarks/lm') {
      return json(landmarkDetail({
        name: 'Tower',
        contributors: [{ user_id: 'u' }, { user_id: 'amy' }],
        models: [{ ...model('m0', 'face', 0), created_at: oct21 }, model('m1', 'face', 1)],
        jobs: [
          FAILED_FULL,
          job({ id: 'jw', kind: 'face', sector: 6, status: 'failed', error: 'only 2 of 6 photos could be registered; need at least 3 overlapping views' }),
          job({ id: 'j0', kind: 'face', sector: 0, status: 'done', created_at: now - 900, started_at: now - 800, finished_at: now - 300 }),
        ],
      }));
    }
    if (url === '/api/landmarks/lm/poses') return json(posesDoc({ as_of: now - 300 }));
    if (url === '/api/contributors') {
      if (registerFails) throw new TypeError('Failed to fetch');
      return json({ nickname: JSON.parse(init.body).nickname, terms: 'terms-v1', contributor_token: TOKEN }, 201);
    }
    if (url === '/api/contributors/accept') return json({ nickname: 'u', terms: 'terms-v1' });
    if (url === '/api/photos') {
      if (refusal) return json({ detail: [{ type: refusal, loc: ['body', 'contributor_token'], msg: 'x', input: null }] }, 422);
      return json({
        results: [
          { ...uploadResult(1, true, 'e'.repeat(64)), distance_m: 50 },
          { filename: 'IMG_2.jpg', photo_id: null, accepted: false, reasons: ['no_gps', 'too_far'], warnings: [], landmark_id: null, sector: null, bearing_deg: null, distance_m: null },
        ],
        triggered_jobs: [{ kind: 'face', sector: 1 }, { kind: 'full', sector: null }],
      });
    }
    if (url === '/api/photos/p8/share_location') return json({ detail: 'unknown photo' }, 404);
    return json({ detail: 'Not Found' }, 404);
  };
  await inLang('en', async () => {
    await openWithRequests(handler, { restored: false });
    assert.equal(doc.title, 'Tower · Crowd3D');
    assert.equal(text(doc.getElementById('lm-meta')), '43.77300, 11.25470 · Check-in radius 150 m · 34 photos · 2 contributors');
    assert.equal(text(doc.getElementById('lm-progress-text')), 'Coverage 80%');
    assert.equal(
      text(doc.getElementById('lm-next')),
      'The full model could not be built: The photos overlap too little to find a pair to start the reconstruction from. ' +
        'Keep uploading photos: it will be retried automatically once there are enough; or press "Rebuild" to retry now with the 34 photos already here.',
    );
    // the compass: its summary, a wedge's tooltip, the short names around it
    const compass = doc.getElementById('compass').innerHTML;
    assert.match(compass, /aria-label="Coverage compass: enough photos on 7 of 8 sides, 34 photos in all"/);
    assert.match(compass, /<title>West \(W\): 4 photos \(4 needed\)\. Enough photos\. Facade build failed\.<\/title>/);
    assert.match(compass, />NE<\/text>/);
    assert.match(compass, />photos<\/text>/);
    assert.match(text(doc.getElementById('coverage-rule')), /^Photos are grouped into 8 sides/);
    // a failed job: why, how it gets retried, and the raw error under its label
    const west = doc.getElementById('job-list').querySelectorAll('li')[1];
    assert.match(text(west), /^West face/);
    assert.match(text(west), /Only 2 of 6 photos could be matched with each other/);
    assert.match(text(west), /It will be retried automatically once this side has a few more photos\./);
    assert.equal(text(west.querySelector('summary')), 'Technical details');
    assert.equal(west.querySelector('code').textContent, 'only 2 of 6 photos could be registered; need at least 3 overlapping views');
    const done = doc.getElementById('job-list').querySelectorAll('li')[2];
    assert.equal(text(done.querySelector('.job-title')), 'North face');
    assert.equal(text(done.querySelector('.job-meta')), '5 photos · Finished 5 minutes ago, took 500 seconds');
    // models: a date in the page language
    const meta = doc.getElementById('model-list').querySelectorAll('.model-title');
    assert.deepEqual(meta[0].querySelectorAll('span').map((x) => plainSpaces(text(x))), ['North face', 'Oct 21, 9:05 AM']);
    assert.match(text(doc.getElementById('model-list')), /1,000 points · 5 of 5 photos located · Georeferenced \(GPS alignment residual 3\.0 m\) · Heading error 4\.0°/);
    // the public poses: counts, rows, sources and the measured coverage
    assert.equal(text(doc.getElementById('poses-as-of')), 'Updated 5 minutes ago');
    assert.equal(
      text(doc.getElementById('poses-count')),
      'Public: 4 of 9 photos, 3 shown; 1 not shown (position and heading could not be determined reliably).',
    );
    const rows = doc.getElementById('pose-rows').querySelectorAll('tr').map((tr) => tr.querySelectorAll('td').map(text));
    assert.deepEqual(rows, [
      ['3D reconstruction', '12 m east, 3 m south', '183° (S)', '±16 m · ±21°'],
      ['Phone GPS + compass (unverified)', '40 m west, 7 m north', '92° (E)', '±41 m · ±31°'],
      ['3D reconstructionYours', '0 m east, 25 m north', '359° (N)', '±16 m · ±21°'],
    ]);
    assert.match(text(doc.getElementById('poses-coverage')), /^± ranges are 2σ\. .*3D reconstruction: 146 of 165 \(88%\) on 4 test landmarks over 2 runs; /);
    assert.match(text(doc.getElementById('poses-sources')), /^3D reconstruction: The photo was used to build the 3D model/);
    // the short EXIF label of the table; its hint says what was not checked
    assert.match(
      text(doc.getElementById('poses-sources')),
      / Phone GPS \+ compass \(unverified\): The location and compass heading the phone saved with the photo, not checked against the model; the range is much wider\./,
    );
    assert.deepEqual(doc.getElementById('contributors').querySelectorAll('li').map(text), ['amy', 'u (you)']);
    // your photos on this device
    const own = doc.getElementById('shared-own');
    assert.match(text(own), /^You shared 1 photo from this device\. Once you withdraw it, others can no longer see it or where it was taken\./);
    assert.equal(own.querySelector('ul').getAttribute('aria-label'), 'Your shared photos');

    // upload, as the registered nickname: accepted and rejected photos, then the server refusing for the registration
    assert.equal(text(doc.getElementById('upload-as')), 'Uploading as u');
    assert.equal(doc.getElementById('upload-form').hidden, false);
    assert.equal(doc.getElementById('register-form').hidden, true);
    const form = doc.getElementById('upload-form');
    const button = new FakeElement('button');
    button.setAttribute('type', 'submit');
    form.append(button);
    const status = () => text(doc.getElementById('upload-status'));
    // the photo picker speaks the page language (the browser's own file input text is hidden)
    const chosen = () => text(doc.getElementById('files-chosen'));
    assert.equal(chosen(), 'No photos chosen');
    for (const [n, want] of [[1, '1 photo chosen'], [3, '3 photos chosen'], [0, 'No photos chosen']]) {
      doc.getElementById('files').files = Array.from({ length: n }, (_, i) => new File(['x'], `IMG_${i}.jpg`, { type: 'image/jpeg' }));
      doc.getElementById('files').dispatch('change');
      assert.equal(chosen(), want);
    }
    const submit = async (n = 2) => {
      doc.getElementById('files').files = Array.from({ length: n }, (_, i) => new File(['x'], `IMG_${i}.jpg`, { type: 'image/jpeg' }));
      doc.getElementById('files').dispatch('change');
      form.dispatch('submit');
      await settle();
    };
    await submit(0);
    assert.equal(status(), 'Please choose the photos to upload.');
    await submit(51);
    assert.equal(status(), 'You can upload up to 50 photos at a time; you chose 51.');
    await submit();
    // the builds inside the sentence: lower case and with articles, unlike the titles at the start of a line
    assert.equal(status(), '1 photo accepted, 1 rejected. Now building: the north-east face and the full model.');
    assert.equal(button.textContent, 'Upload');
    const results = doc.getElementById('upload-results');
    assert.deepEqual(results.querySelectorAll('th').map(text), ['Photo', 'Result']);
    const [ok, bad] = results.querySelectorAll('tbody tr').map((tr) => tr.querySelectorAll('td').map(text));
    assert.deepEqual(ok, ['IMG_1.jpgNorth · 50 m', 'AcceptedPublic once its position and heading are known, at the earliest after the next build (you can withdraw it)']);
    // sent: the picker is emptied
    assert.equal(doc.getElementById('files').value, '');
    assert.deepEqual(bad, ['IMG_2.jpg', 'RejectedNo GPS location. Too far from the landmark']);
    assert.match(text(own), /^You shared 2 photos from this device\. Once you withdraw them/);

    // the server no longer knows the registration: register again; the results above go (D33, a screenshot round)
    const resultRows = () => results.querySelectorAll('tbody tr').length;
    const registerForm = doc.getElementById('register-form');
    refusal = 'register_required';
    await submit();
    assert.equal(status(), "The server no longer knows this device's registration. Please register again.");
    assert.equal(results.hasChildNodes(), false);
    assert.equal(registerForm.hidden, false);
    assert.equal(doc.getElementById('nickname-field').hidden, false);
    assert.equal(text(doc.getElementById('register-intro')), 'Before your first upload, register a nickname and accept the Terms of Use. You only register once; this device remembers it.');
    assert.equal(text(doc.getElementById('register-submit')), 'Register');
    assert.equal(doc.getElementById('nickname').value, 'u');
    registerForm.dispatch('submit');
    await settle();
    assert.equal(status(), 'Please read and accept the Terms of Use first.');
    doc.getElementById('agree-terms').checked = true;
    doc.getElementById('nickname').value = ' ';
    registerForm.dispatch('submit');
    await settle();
    assert.equal(status(), 'Please enter a nickname (up to 64 characters).');
    doc.getElementById('nickname').value = 'u';
    registerFails = true;
    registerForm.dispatch('submit');
    await settle();
    assert.equal(status(), 'Registration failed: Could not reach the server. Check your connection and try again.');
    registerFails = false;
    registerForm.dispatch('submit');
    await settle();
    assert.equal(status(), 'Registered. You can upload photos now.');
    assert.equal(text(doc.getElementById('upload-as')), 'Uploading as u');
    refusal = null;
    await submit();
    assert.equal(resultRows(), 2);

    // a new agreement version: accept it once, no new nickname; the results above go
    refusal = 'terms_outdated';
    await submit();
    assert.equal(status(), 'Not uploaded yet: accept the new Terms of Use first; the photos you chose stay selected.');
    assert.equal(results.hasChildNodes(), false);
    assert.equal(registerForm.hidden, false);
    assert.equal(doc.getElementById('nickname-field').hidden, true);
    assert.equal(text(doc.getElementById('upload-as')), 'Registered as u');
    assert.equal(text(doc.getElementById('register-intro')), 'The Terms of Use have changed. Read and accept the new version to keep uploading; your nickname stays the same.');
    assert.equal(text(doc.getElementById('register-submit')), 'Accept the new version');
    doc.getElementById('agree-terms').checked = true;
    registerForm.dispatch('submit');
    await settle();
    assert.equal(status(), 'You accepted the new Terms of Use and can upload again.');
    assert.equal(doc.getElementById('upload-form').hidden, false);
    // any other refusal: the upload failed, and the results of the upload before are gone too
    refusal = null;
    await submit();
    assert.equal(resultRows(), 2);
    refusal = 'terms_required';
    await submit();
    assert.equal(status(), 'Upload failed: Please read and accept the Terms of Use first.');
    assert.equal(results.hasChildNodes(), false);

    // forgetting this device's registration: the photos shared from it can still be withdrawn; the results above,
    // and the nickname typed when registering, go
    refusal = null;
    await submit();
    assert.equal(resultRows(), 2);
    assert.equal(doc.getElementById('nickname').value, 'u');
    doc.getElementById('forget-registration').click();
    await settle();
    assert.equal(status(), "This device's registration is forgotten. You can still withdraw the photos you shared from this device below.");
    assert.equal(results.hasChildNodes(), false);
    assert.equal(doc.getElementById('nickname').value, '');
    assert.equal(storage.getItem(CONTRIBUTOR_KEY), null);
    assert.deepEqual(doc.getElementById('contributors').querySelectorAll('li').map(text), ['amy', 'u']);

    // withdrawing: one done, one the server no longer has
    refusal = null;
    const all = own.querySelectorAll('button').find((b) => text(b) === 'Withdraw all');
    assert.ok(all, 'Withdraw all');
    assert.equal(own.querySelectorAll('button').filter((b) => text(b) === 'Withdraw').length, 2);
    globalThis.fetch = async (url, init) => (url === '/api/photos/p1/share_location' ? json({ photo_id: 'p1', share_location: false }) : handler(url, init));
    all.click();
    await settle();
    assert.equal(text(own), 'Withdrew 1 photo. 1 photo could not be withdrawn: the server no longer has it.');

    assertNoChinese(doc.body, 'landmark page');
    assert.ok(!CJK.test(doc.title));
  });

  // The same upload in Chinese: the two sentences of the summary join without a space (common.sentenceSep), the
  // builds listed with 、 (D33, D32 leftover)
  storage.setItem(CONTRIBUTOR_KEY, JSON.stringify({ token: TOKEN, nickname: 'u', terms: 'terms-v1' }));
  refusal = null;
  await openWithRequests(handler, { restored: false });
  const form = doc.getElementById('upload-form');
  const button = new FakeElement('button');
  button.setAttribute('type', 'submit');
  form.append(button);
  doc.getElementById('files').files = [new File(['x'], 'IMG_0.jpg', { type: 'image/jpeg' }), new File(['x'], 'IMG_1.jpg', { type: 'image/jpeg' })];
  form.dispatch('submit');
  await settle();
  const status = text(doc.getElementById('upload-status'));
  assert.equal(status, '1 张通过，1 张未通过。已开始生成：东北立面、完整模型。');
  assert.doesNotMatch(status, /。 /, 'no Latin space after a Chinese full stop');
});

test('the registration texts, the agreement box and its hint in English keep every statement of the Chinese (D32, D33)', () => {
  inLang('en', () => {
    assert.equal(`${t('register.terms.pre')}${t('register.terms.link')}${t('register.terms.post')}`, 'I have read and accept the Terms of Use.');
    const hint = t('register.terms.hint');
    for (const part of [
      /^Your photos are shown publicly/,
      /EXIF and other capture data removed/,
      /at most 1600\u00a0px on the longer side/,
      /where they were taken \(rounded to 1\u00a0m\)/,
      /heading \(rounded to 1°\)/,
      /field of view \(rounded to 1°\)/,
      /field of view \(rounded to 1°\); only photos whose position and heading can be worked out are shown\. Your nickname/, // D33
      /Your nickname appears among the contributors of each landmark where a photo of yours is accepted, but not which photo anyone uploaded or when/,
      /only appear after that landmark's next build has finished/,
      // the remaining risk (D30 owner question 2)
      /anyone can make that build start soon with a manual rebuild or by uploading a few photos/,
      /someone could match the photos that appear then with your nickname and guess which ones are yours/,
      /this is easier when few people upload there/,
      /If you do not accept, you cannot register, and so cannot upload\./, // accepted at registration (D33)
      /You only register once\./,
      /After uploading, you can withdraw your photos on this device\.$/,
    ]) {
      assert.match(hint, part);
    }
    assert.doesNotMatch(hint, /you cannot upload;|rounded to 5°/);
    assert.ok(!CJK.test(hint));
    assert.equal(t('register.intro'), 'Before your first upload, register a nickname and accept the Terms of Use. You only register once; this device remembers it.');
    assert.equal(t('register.forget'), "Forget this device's registration");
    assert.equal(t('upload.as', { name: 'Kim' }), 'Uploading as Kim');
    assert.equal(t('api.register.required'), 'Please register a nickname and accept the Terms of Use first.');
  });
  // this device, as the withdrawal texts and the agreement's section 6 say (not "this browser")
  for (const key of ['register.intro', 'register.lost', 'register.forget', 'register.forgotten']) {
    assert.match(t(key), /这台设备/, key);
    inLang('en', () => assert.match(t(key), /this device/i, key));
  }
});

test('the poses hint and the upload result say that only photos with a determined position and heading are public, in both languages (D33)', () => {
  assert.equal(t('poses.hint'), '通过检查、能确定位置和朝向的照片都会公开（撤回的除外）：位置取整到 1\u00a0米，朝向和视角取整到 1°，不显示上传者和时间。每次建模结束后更新。');
  assert.equal(t('upload.public'), '确定了位置和朝向后公开，最早在下次建模结束后（可撤回）');
  inLang('en', () => {
    assert.equal(
      t('poses.hint'),
      'Every accepted photo whose position and heading can be determined is public unless withdrawn: positions are rounded to 1\u00a0m, headings and fields of view to 1°, without the uploader or the time. Updated after each build.',
    );
    assert.equal(t('upload.public'), 'Public once its position and heading are known, at the earliest after the next build (you can withdraw it)');
  });
});

// The last test of this file: every key the pages asked for while the tests above ran has a text.
test('the pages asked for no text strings.js does not have (D32)', () => {
  assert.deepEqual([...missing], []);
});
