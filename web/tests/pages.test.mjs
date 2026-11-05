// Run from the repo root: node --test 'web/tests/*.test.mjs'
// Imports the real page scripts against the fake DOM in fake-dom.mjs, with fetch and Leaflet mocked.
import assert from 'node:assert/strict';
import { afterEach, mock, test } from 'node:test';

import { FakeElement, installDom, settle, text } from './fake-dom.mjs';

const realFetch = globalThis.fetch;
const realWarn = console.warn;
let doc = null;
let page = 0;

afterEach(() => {
  mock.timers.reset();
  globalThis.fetch = realFetch;
  console.warn = realWarn;
  delete globalThis.L;
  delete globalThis.localStorage;
  delete globalThis.crowd3dPrefs;
  doc?.restore();
  doc = null;
});

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

function fakeLeaflet() {
  const tooltips = [];
  const layer = () => {
    const o = {
      addTo: () => o,
      setView: () => o,
      on: () => o,
      bindPopup: () => o,
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
    L: { map: layer, tileLayer: layer, layerGroup: layer, circle: layer, circleMarker: layer, latLngBounds: () => ({}) },
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
    globalThis.crowd3dPrefs = { langChoice: () => 'en', setLang: (c) => calls.push(c) };
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
  doc.getElementById('user-id').value = 'u';
  doc.getElementById('agree-terms').checked = true;
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
 * agreement box again when it restored the form after a reload. */
async function openWithRequests(handler, { restored = true } = {}) {
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

test('landmark page: no upload without the user agreement; the box is unticked on every load, stays ticked within the page view, and a photo can be withdrawn (D32)', async () => {
  const storage = fakeStorage();
  globalThis.localStorage = storage;
  const token = 'b'.repeat(64);
  let n = 0;
  const handler = (url, init) => {
    if (url === '/api/landmarks/lm') return json(landmarkDetail());
    if (url === '/api/photos') return json({ results: [uploadResult(++n, init.body.get('terms') === 'terms-v1', token)], triggered_jobs: [] });
    if (url === '/api/photos/p1/share_location') return json({ photo_id: 'p1', share_location: false });
    return json({ detail: 'Not Found' }, 404);
  };
  let requests = await openWithRequests(handler);
  const box = () => doc.getElementById('agree-terms');
  assert.equal(box().checked, false, 'unticked although the browser restored it');
  assert.equal(doc.getElementById('shared-own').hidden, true);

  const form = doc.getElementById('upload-form');
  const button = new FakeElement('button');
  button.setAttribute('type', 'submit');
  form.append(button);
  doc.getElementById('user-id').value = 'u';
  const uploads = () => requests.filter((r) => r.url === '/api/photos');
  const submit = async () => {
    doc.getElementById('files').files = [new File(['x'], 'IMG.jpg', { type: 'image/jpeg' })];
    form.dispatch('submit');
    await settle();
  };

  // unticked: nothing is sent, the message says why, and the box gets the focus
  let focused = 0;
  box().focus = () => focused++;
  await submit();
  assert.equal(uploads().length, 0);
  assert.equal(text(doc.getElementById('upload-status')), '请先阅读并同意《用户协议》');
  assert.equal(focused, 1);

  box().checked = true;
  await submit();
  let body = uploads().at(-1).init.body;
  assert.deepEqual(body.getAll('terms'), ['terms-v1']);
  assert.equal(body.has('share_location'), false);
  assert.equal(box().checked, true, 'stays ticked for further uploads in this page view');
  assert.match(text(doc.getElementById('upload-results')), /通过已公开（可撤回）/);
  assert.deepEqual(JSON.parse(storage.getItem(SHARED_KEY)), [
    { photo_id: 'p1', token, filename: 'IMG_1.jpg', public_id: 'pub100000000' },
  ]);
  let own = doc.getElementById('shared-own');
  assert.equal(own.hidden, false);
  assert.match(text(own), /你在这台设备上公开了 1 张照片。撤回后，别人就看不到这些照片和它们的拍摄位置了。/);
  assert.match(text(own), /IMG_1\.jpg/);
  await submit();                                   // a second upload in the same page view needs no new tick
  assert.equal(uploads().length, 2);
  assert.deepEqual(uploads().at(-1).init.body.getAll('terms'), ['terms-v1']);

  // A reload: unticked again, the list of shared photos comes back from this browser
  requests = await openWithRequests(handler);
  storage.setItem(SHARED_KEY, JSON.stringify([{ photo_id: 'p1', token, filename: 'IMG_1.jpg', public_id: 'pub100000000' }]));
  requests = await openWithRequests(handler);
  assert.equal(box().checked, false);
  own = doc.getElementById('shared-own');
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

test('landmark page: an upload the server refuses for the agreement shows why (D32)', async () => {
  globalThis.localStorage = fakeStorage();
  const refusal = (type) => ({
    detail: [{ type, loc: ['body', 'terms'], msg: 'uploading needs agreeing to the user agreement terms-v1 (/terms.html)',
      input: 'terms-v0', ctx: { version: 'terms-v1', url: '/terms.html' } }],
  });
  let type = 'terms_outdated';
  await openWithRequests(
    (url) => {
      if (url === '/api/landmarks/lm') return json(landmarkDetail());
      if (url === '/api/photos') return json(refusal(type), 422);
      return json({ detail: 'Not Found' }, 404);
    },
    { restored: false },
  );
  const form = doc.getElementById('upload-form');
  const button = new FakeElement('button');
  button.setAttribute('type', 'submit');
  form.append(button);
  doc.getElementById('user-id').value = 'u';
  doc.getElementById('agree-terms').checked = true;
  for (const [t, message] of [['terms_outdated', '用户协议已更新，请刷新页面后重新阅读并同意'], ['terms_required', '请先阅读并同意《用户协议》']]) {
    type = t;
    doc.getElementById('files').files = [new File(['x'], 'IMG.jpg', { type: 'image/jpeg' })];
    form.dispatch('submit');
    await settle();
    assert.equal(text(doc.getElementById('upload-status')), `上传失败：${message}`);
  }
  assert.equal(doc.getElementById('shared-own').hidden, true);
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

test('the user agreement: one version in records.py, api.js and both language versions of terms.html, which say what is public, what is not and how to withdraw (D32)', async () => {
  // The agreement version uploads record (crowd3d/records.py TERMS_CURRENT) stands for exactly the text of
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
  for (const part of ['草案，待法律审核', '1 米', '1°', '5°', '1600', 'EXIF', '撤回', '可辨认的人',
    // never shown
    '哪张照片是谁上传的', '上传时间和拍摄时间', '内部的编号', '原始文件', '每个人上传了多少张', '没有通过检查的照片',
    // the remaining linking risk of the upload hint (tests/test_api.py
    // test_anyone_can_move_the_poses_snapshot_soon_after_a_new_upload_as_the_agreement_says)
    '任何人都能用手动重建或者上传几张照片让下一次建模很快开始']) {
    assert.ok(plain(zh).includes(part), part);
  }
  for (const part of ['Draft - pending legal review', '1 m', '1°', '5°', '1600 px', 'EXIF', 'withdraw', 'Withdraw all',
    'who uploaded which photo', 'when a photo was uploaded or taken', 'internal photo IDs', 'the original file',
    'how many photos each person uploaded', 'anyone can make that build start soon']) {
    assert.ok(plain(en).includes(part), part);
  }
  assert.doesNotMatch(plain(en), /[\u4e00-\u9fff]/, 'the English version holds no Chinese');
  for (const a of articles) assert.match(a.body, /<mark>/, 'placeholders are marked');
});

test('landmark page: the agreement box links the agreement, is not ticked in the markup, and its hint names what is public, the rounding, the remaining risk and withdrawal (D32)', async () => {
  const { readFileSync } = await import('node:fs');
  const html = readFileSync(new URL('../landmark.html', import.meta.url), 'utf8');
  const label = html.match(/<label class="check"><input id="agree-terms"[^>]*>[\s\S]*?<\/label>/);
  assert.ok(label, 'the agreement checkbox and its label');
  assert.match(label[0], /type="checkbox"/);
  assert.doesNotMatch(label[0], /\bchecked\b/, 'not ticked in the markup');
  assert.doesNotMatch(label[0], /\brequired\b/, 'the page shows its own message');
  assert.match(label[0], /<span>我已阅读并同意<\/span><a href="terms\.html" target="_blank" rel="noopener">《用户协议》<\/a>/);
  assert.doesNotMatch(html, /share-location|share_location/);
  const hint = html.match(/<p id="agree-terms-hint"[^>]*>([^<]*)<\/p>/);
  assert.ok(hint, 'the agreement hint');
  for (const part of [
    /^上传的照片会公开显示/,
    /去掉 EXIF 等拍摄信息/,
    /长边不超过 1600 像素/,
    /拍摄位置（取整到 1 米）/,
    /朝向（取整到 1°）/,
    /视角（取整到 5°）/,
    /你的昵称会出现在共同完成者里/,
    /不显示哪张照片是谁传的、什么时候传的/,
    /公开的照片和位置要等这里下一次建模结束后才显示/, // the public poses are a snapshot (D30, owner question 2)
    // the remaining risk (owner question 2): anyone can end the snapshot soon (tests/test_api.py
    // test_anyone_can_move_the_poses_snapshot_soon_after_a_new_upload_as_the_agreement_says)
    /任何人都能用手动重建或者上传几张照片让下一次建模很快开始/,
    /别人可能把那时新出现的照片和你的昵称对上，猜出哪些是你拍的/,
    /这里上传的人少时更容易/,
    /不同意就不能上传/,
    /上传后可以在这台设备上撤回/,
  ]) {
    assert.match(hint[1], part);
  }
  for (const page of ['index.html', 'landmark.html']) {
    const markup = readFileSync(new URL(`../${page}`, import.meta.url), 'utf8');
    assert.match(markup, /<footer class="site-footer">[\s\S]*<a href="terms\.html">用户协议<\/a>/, page);
  }
});

test('landmark page: contributors are nicknames by name without counts, even from a server that still sends counts (D30)', async () => {
  globalThis.localStorage = fakeStorage({ 'crowd3d.user_id': 'zoe' });
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
    rounding: { position_m: 1, angle_deg: 1, fov_deg: 5, aspect_decimals: 2 },
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
