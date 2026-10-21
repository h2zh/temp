// Run from the repo root: node --test 'web/tests/*.test.mjs'
import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

import {
  ApiError,
  REASON_TEXT,
  WARNING_TEXT,
  codeText,
  createLandmark,
  errorMessage,
  getLandmark,
  isActiveJob,
  listLandmarks,
  rebuild,
  uploadPhotos,
} from '../api.js';
import { fmtDeg, fmtMeters, fmtPercent, fmtTime } from '../ui.js';

const realFetch = globalThis.fetch;
let calls = [];

function mockFetch(status, body, { raw = false } = {}) {
  calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init });
    return new Response(raw ? body : JSON.stringify(body), { status });
  };
}

afterEach(() => {
  globalThis.fetch = realFetch;
});

test('every reason and warning code from the spec has its Chinese text', () => {
  const expected = {
    unreadable_image: '无法读取图片',
    no_gps: '缺少GPS定位',
    no_landmark: '附近没有地标',
    too_far: '距离地标太远',
    facing_away: '镜头没有朝向建筑',
    too_small: '分辨率太低',
    too_blurry: '照片太模糊',
    duplicate: '重复照片',
    too_large: '文件过大',
    no_heading: '缺少指南针朝向',
    magnetic_heading: '使用磁北朝向',
  };
  for (const [code, text] of Object.entries(expected)) assert.equal(codeText(code), text, code);
  assert.equal(Object.keys(REASON_TEXT).length + Object.keys(WARNING_TEXT).length, Object.keys(expected).length);
  assert.equal(codeText('something_new'), 'something_new');
});

test('errorMessage handles FastAPI string and validation-list details', () => {
  assert.equal(errorMessage(409, { detail: 'landmark exists' }), 'landmark exists');
  assert.equal(
    errorMessage(422, { detail: [{ loc: ['body', 'lat'], msg: 'too big' }, { loc: ['body', 'name'], msg: 'empty' }] }),
    '纬度：too big；名称：empty',
  );
  assert.equal(errorMessage(500, null), '请求失败（HTTP 500）');
});

test('listLandmarks unwraps the envelope; getLandmark escapes the id', async () => {
  mockFetch(200, { landmarks: [{ id: 'a' }] });
  assert.deepEqual(await listLandmarks(), [{ id: 'a' }]);
  assert.equal(calls[0].url, '/api/landmarks');

  mockFetch(200, { id: 'a b' });
  await getLandmark('a b/c');
  assert.equal(calls[0].url, '/api/landmarks/a%20b%2Fc');
});

test('createLandmark posts JSON and surfaces HTTP errors as ApiError', async () => {
  mockFetch(201, { id: 'x' });
  await createLandmark({ name: '塔', lat: 1, lon: 2 });
  assert.equal(calls[0].init.method, 'POST');
  assert.equal(calls[0].init.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(calls[0].init.body), { name: '塔', lat: 1, lon: 2 });

  mockFetch(409, { detail: 'duplicate id' });
  await assert.rejects(createLandmark({ name: 'x', lat: 0, lon: 0 }), (err) => {
    assert.ok(err instanceof ApiError);
    assert.equal(err.status, 409);
    assert.equal(err.message, 'duplicate id');
    return true;
  });
});

test('non-JSON error bodies and network failures become readable ApiErrors', async () => {
  mockFetch(502, '<html>bad gateway</html>', { raw: true });
  await assert.rejects(listLandmarks(), (err) => err.status === 502 && err.message === '请求失败（HTTP 502）');

  globalThis.fetch = async () => {
    throw new TypeError('Failed to fetch');
  };
  await assert.rejects(listLandmarks(), (err) => err instanceof ApiError && err.status === 0);
});

test('uploadPhotos sends multipart with field names from the contract', async () => {
  mockFetch(200, { results: [], triggered_jobs: [] });
  const files = [new File(['a'], 'one.jpg', { type: 'image/jpeg' }), new File(['b'], 'two.jpg', { type: 'image/jpeg' })];
  await uploadPhotos({ files, userId: '小明', landmarkId: 'demo-building' });
  const form = calls[0].init.body;
  assert.equal(calls[0].url, '/api/photos');
  assert.ok(form instanceof FormData);
  assert.deepEqual(
    form.getAll('files').map((f) => f.name),
    ['one.jpg', 'two.jpg'],
  );
  assert.equal(form.get('user_id'), '小明');
  assert.equal(form.get('landmark_id'), 'demo-building');
  assert.equal(calls[0].init.headers['Content-Type'], undefined, 'browser must set the multipart boundary');

  mockFetch(200, { results: [], triggered_jobs: [] });
  await uploadPhotos({ files, userId: 'u' });
  assert.equal(calls[0].init.body.has('landmark_id'), false);
});

test('rebuild omits sector for full models', async () => {
  mockFetch(202, { id: 'j' });
  await rebuild('lm', 'full');
  assert.equal(calls[0].url, '/api/landmarks/lm/rebuild');
  assert.deepEqual(JSON.parse(calls[0].init.body), { kind: 'full' });

  mockFetch(202, { id: 'j' });
  await rebuild('lm', 'face', 0);
  assert.deepEqual(JSON.parse(calls[0].init.body), { kind: 'face', sector: 0 });
});

test('isActiveJob', () => {
  assert.ok(isActiveJob({ status: 'queued' }));
  assert.ok(isActiveJob({ status: 'running' }));
  assert.ok(!isActiveJob({ status: 'done' }));
  assert.ok(!isActiveJob({ status: 'failed' }));
});

test('formatters', () => {
  assert.equal(fmtPercent(0.625), '63%');
  assert.equal(fmtPercent(2), '100%');
  assert.equal(fmtMeters(3.14), '3.1 米');
  assert.equal(fmtMeters(42.6), '43 米');
  assert.equal(fmtMeters(null), '—');
  assert.equal(fmtDeg(359.7), '0°');
  assert.equal(fmtDeg(null), '—');
  assert.equal(fmtTime(1000, 1030), '刚刚');
  assert.equal(fmtTime(1000, 1000 + 5 * 60), '5 分钟前');
  assert.equal(fmtTime(1000, 1000 + 3 * 3600), '3 小时前');
  assert.match(fmtTime(1000, 1000 + 3 * 86400), /^\d+月\d+日 \d\d:\d\d$/);
});
