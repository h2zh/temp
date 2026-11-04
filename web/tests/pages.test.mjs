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
    label: '素模（照片未对齐）',
  };
  await openLandmark({
    '/api/landmarks/lm': landmarkDetail({
      models: [
        { ...model('m0', 'face', 0), stylized: refused },
        { ...model('m1', 'face', 1), stylized: { status: 'pending', url: null, registration: null, label: null } },
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

test('landmark page: no per-photo data, even from a server that still sends it, and no camera requests (D30)', async () => {
  // An older server's detail listed each photo with its uploader, position and the SfM camera's exact pose, and a
  // model could name a cameras_url; the page reads neither.
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
    ['/api/landmarks/lm'],
  );
  const ids = doc.body.children.map((n) => n.getAttribute('id'));
  for (const id of ['photos-title', 'photo-rows', 'show-cameras']) assert.ok(!ids.includes(id), id);
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
    manage_token: shared ? token : null,
  };
}

/** Opens the landmark page with a fetch mock that records each request's init; restored is a browser that ticked the
 * consent box again when it restored the form after a reload. */
async function openWithRequests(handler, { restored = true } = {}) {
  doc?.restore();
  doc = installDom({ search: '?id=lm' });
  console.warn = () => {};
  if (restored) doc.getElementById('share-location').checked = true;
  const requests = [];
  globalThis.fetch = async (url, init = {}) => {
    requests.push({ url, init });
    return handler(url, init);
  };
  await import(`../landmark.js?page=${++page}`);
  await settle();
  return requests;
}

test('landmark page: the location consent box is unticked on every load, sent only when ticked, and a shared photo can be withdrawn (D30)', async () => {
  const storage = fakeStorage();
  globalThis.localStorage = storage;
  const token = 'b'.repeat(64);
  let n = 0;
  const handler = (url, init) => {
    if (url === '/api/landmarks/lm') return json(landmarkDetail());
    if (url === '/api/photos') return json({ results: [uploadResult(++n, init.body.get('share_location') === 'true', token)], triggered_jobs: [] });
    if (url === '/api/photos/p2/share_location') return json({ photo_id: 'p2', share_location: false });
    return json({ detail: 'Not Found' }, 404);
  };
  let requests = await openWithRequests(handler);
  const box = () => doc.getElementById('share-location');
  assert.equal(box().checked, false, 'unticked although the browser restored it');
  assert.equal(doc.getElementById('shared-own').hidden, true);

  const form = doc.getElementById('upload-form');
  const button = new FakeElement('button');
  button.setAttribute('type', 'submit');
  form.append(button);
  doc.getElementById('user-id').value = 'u';
  const upload = async (tick) => {
    box().checked = tick;
    doc.getElementById('files').files = [new File(['x'], 'IMG.jpg', { type: 'image/jpeg' })];
    form.dispatch('submit');
    await settle();
    return requests.filter((r) => r.url === '/api/photos').at(-1).init.body;
  };

  let body = await upload(false);
  assert.equal(body.has('share_location'), false);
  assert.match(text(doc.getElementById('upload-results')), /通过位置不公开/);
  assert.equal(storage.getItem(SHARED_KEY), null);
  assert.equal(doc.getElementById('shared-own').hidden, true);

  body = await upload(true);
  assert.deepEqual(body.getAll('share_location'), ['true']);
  assert.equal(box().checked, false, 'consent is given per upload');
  assert.match(text(doc.getElementById('upload-results')), /通过位置公开/);
  assert.deepEqual(JSON.parse(storage.getItem(SHARED_KEY)), [{ photo_id: 'p2', token, filename: 'IMG_2.jpg' }]);
  let own = doc.getElementById('shared-own');
  assert.equal(own.hidden, false);
  assert.match(text(own), /你在这台设备上公开了 1 张照片的拍摄位置/);
  assert.match(text(own), /IMG_2\.jpg/);

  // A reload: unticked again, the list of shared photos comes back from this browser
  requests = await openWithRequests(handler);
  assert.equal(box().checked, false);
  own = doc.getElementById('shared-own');
  assert.equal(own.hidden, false);
  const withdraw = own.querySelectorAll('button').find((b) => b.textContent === '撤回');
  assert.equal(own.querySelectorAll('button').some((b) => b.textContent === '全部撤回'), false, 'one photo: one button');
  withdraw.click();
  await settle();
  const post = requests.find((r) => r.url === '/api/photos/p2/share_location');
  assert.equal(post.init.method, 'POST');
  assert.deepEqual(JSON.parse(post.init.body), { share: false, token });
  assert.equal(storage.getItem(SHARED_KEY), null);
  assert.equal(text(own), '已撤回 1 张照片的位置公开。');
});

test('landmark page: withdrawing all keeps only the photos whose withdrawal can still succeed (D30)', async () => {
  const shared = ['p1', 'p2', 'p3'].map((id) => ({ photo_id: id, token: id.repeat(32).slice(0, 64), filename: `${id}.jpg` }));
  const storage = fakeStorage({ [SHARED_KEY]: JSON.stringify([...shared, { photo_id: 7 }, null]) });
  globalThis.localStorage = storage;
  await openWithRequests(
    (url) => {
      if (url === '/api/landmarks/lm') return json(landmarkDetail());
      if (url === '/api/photos/p1/share_location') return json({ photo_id: 'p1', share_location: false });
      if (url === '/api/photos/p2/share_location') return json({ detail: 'wrong manage token' }, 403);
      throw new TypeError('Failed to fetch'); // p3: the network is down
    },
    { restored: false },
  );
  const own = doc.getElementById('shared-own');
  assert.match(text(own), /你在这台设备上公开了 3 张照片的拍摄位置/);       // malformed entries are dropped
  own.querySelectorAll('button').find((b) => b.textContent === '全部撤回').click();
  await settle();
  assert.deepEqual(JSON.parse(storage.getItem(SHARED_KEY)), [shared[2]]);
  assert.match(text(own), /已撤回 1 张照片的位置公开；1 张服务器已找不到，无法撤回；1 张撤回失败，请稍后再试。$/);
  assert.match(text(own), /公开了 1 张照片的拍摄位置/);
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
