// Run from the repo root: node --test 'web/tests/*.test.mjs'
import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

import {
  ApiError,
  REASON_TEXT,
  TERMS_VERSION,
  WARNING_TEXT,
  codeText,
  createLandmark,
  errorMessage,
  getLandmark,
  getPoses,
  isActiveJob,
  jobErrorText,
  listLandmarks,
  rebuild,
  uploadPhotos,
  uploadReasonText,
  withdrawShareLocation,
} from '../api.js';
import { fmtCount, fmtDeg, fmtMeters, fmtPercent, fmtTime } from '../ui.js';
import { inLang } from './fake-dom.mjs';

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
    no_exif: '没有 EXIF 信息，请上传相机原图（截图、编辑过或经聊天软件发送的图片会丢失）',
    no_gps: '缺少GPS定位',
    no_landmark: '附近没有地标',
    too_far: '距离地标太远',
    facing_away: '镜头没有朝向建筑',
    too_small: '分辨率太低',
    too_blurry: '照片太模糊',
    duplicate: '重复照片',
    too_large: '文件过大',
    not_subject: '离建筑超过 50 米，建筑又不是画面主体：请走近拍，或让建筑占满画面',
    no_heading: '缺少指南针朝向',
    magnetic_heading: '使用磁北朝向',
    subject_unchecked: '没有检查建筑是不是照片主体',
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

test('errorMessage shows its own text for an upload without the current user agreement (D32)', () => {
  const detail = (type) => ({
    detail: [{ type, loc: ['body', 'terms'], msg: 'uploading needs agreeing to the user agreement terms-v1 (/terms.html)',
      input: null, ctx: { version: 'terms-v1', url: '/terms.html' } }],
  });
  assert.equal(errorMessage(422, detail('terms_required')), '请先阅读并同意《用户协议》');
  assert.equal(errorMessage(422, detail('terms_outdated')), '用户协议已更新，请刷新页面后重新阅读并同意');
  // other types keep the field label and the server's message
  assert.equal(errorMessage(422, { detail: [{ type: 'value_error', loc: ['body', 'user_id'], msg: 'x' }] }), '昵称：x');
  assert.equal(errorMessage(422, { detail: [{ type: 'toString', loc: ['body', 'files'], msg: 'y' }] }), '照片：y');
});

test('listLandmarks unwraps the envelope; getLandmark escapes the id', async () => {
  mockFetch(200, { landmarks: [{ id: 'a' }] });
  assert.deepEqual(await listLandmarks(), [{ id: 'a' }]);
  assert.equal(calls[0].url, '/api/landmarks');

  mockFetch(200, { id: 'a b' });
  await getLandmark('a b/c');
  assert.equal(calls[0].url, '/api/landmarks/a%20b%2Fc');
});

test("getPoses asks for the landmark's public photo poses (D30)", async () => {
  mockFetch(200, { format: 'crowd3d.photo-poses', version: 2, photos: [] });
  assert.equal((await getPoses('a b/c')).version, 2);
  assert.equal(calls[0].url, '/api/landmarks/a%20b%2Fc/poses');
  assert.equal(calls[0].init.method, 'GET');
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

  assert.deepEqual(form.getAll('terms'), ['terms-v1']);
  assert.equal(TERMS_VERSION, 'terms-v1');

  mockFetch(200, { results: [], triggered_jobs: [] });
  await uploadPhotos({ files, userId: 'u' });
  assert.equal(calls[0].init.body.has('landmark_id'), false);
  // D32: every upload sends the agreement's version, and never the D30 share_location field
  for (const shareLocation of [false, true, 'true', undefined]) {
    mockFetch(200, { results: [], triggered_jobs: [] });
    await uploadPhotos({ files, userId: 'u', shareLocation });
    assert.deepEqual(calls[0].init.body.getAll('terms'), [TERMS_VERSION], String(shareLocation));
    assert.equal(calls[0].init.body.has('share_location'), false, String(shareLocation));
  }
});

test('withdrawShareLocation posts share false with the manage token (D30)', async () => {
  mockFetch(200, { photo_id: 'p/1', share_location: false });
  const token = 'a'.repeat(64);
  assert.deepEqual(await withdrawShareLocation('p/1', token), { photo_id: 'p/1', share_location: false });
  assert.equal(calls[0].url, '/api/photos/p%2F1/share_location');
  assert.equal(calls[0].init.method, 'POST');
  assert.deepEqual(JSON.parse(calls[0].init.body), { share: false, token });

  mockFetch(403, { detail: 'wrong manage token' });
  await assert.rejects(withdrawShareLocation('p', token), (err) => err instanceof ApiError && err.status === 403);
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

test('formatters in English (D32), and the Chinese ones unchanged around them', () => {
  // local time, as the page shows it
  const oct21 = new Date(2026, 9, 21, 9, 5).getTime() / 1000;
  const later = oct21 + 3 * 86400;
  inLang('en', () => {
    assert.equal(fmtMeters(50), '50 m');
    assert.equal(fmtMeters(3.14), '3.1 m');
    assert.equal(fmtMeters(null), '—');
    assert.equal(fmtCount(1234), '1,234');
    assert.equal(fmtCount(null), '0');
    assert.equal(fmtPercent(0.42), '42%');
    assert.equal(fmtPercent(0.625), '63%');
    assert.equal(fmtDeg(359.7), '0°');
    assert.equal(fmtTime(1000, 1030), 'just now');
    assert.equal(fmtTime(1000, 1000 + 60), '1 minute ago');
    assert.equal(fmtTime(1000, 1000 + 5 * 60), '5 minutes ago');
    assert.equal(fmtTime(1000, 1000 + 3600), '1 hour ago');
    assert.equal(fmtTime(1000, 1000 + 3 * 3600), '3 hours ago');
    // ICU may put a narrow no-break space before AM/PM
    assert.equal(fmtTime(oct21, later).replace(/\u202f/g, ' '), 'Oct 21, 9:05 AM');
  });
  assert.equal(fmtCount(1234), '1,234');
  assert.equal(fmtTime(1000, 1000 + 60), '1 分钟前');
  assert.equal(fmtTime(oct21, later), '10月21日 09:05');
  assert.equal(fmtTime(new Date(2026, 0, 3, 0, 7).getTime() / 1000, later), '1月3日 00:07');
});

test('job errors from the reconstructor are shown in Chinese', () => {
  assert.equal(
    jobErrorText('structure-from-motion found no initial image pair; photos may not overlap enough'),
    '照片之间重叠太少，找不到可以开始重建的照片',
  );
  assert.equal(
    jobErrorText('only 2 of 9 photos could be registered; need at least 3 overlapping views'),
    '9 张照片里只有 2 张能互相对上',
  );
  assert.equal(jobErrorText('need at least 3 photo files, found 1 of 4'), '能用的照片太少');
  assert.equal(jobErrorText('need at least 2 photos'), '能用的照片太少');
  assert.equal(
    jobErrorText('quality check failed, the previous model stays published: only 5 of 9 photos registered (need 60%)'),
    '生成的模型没有通过质量检查，没有发布',
  );
  assert.equal(
    jobErrorText('interrupted 2 times: the process died while reconstructing these photos'),
    '服务器在重建这些照片时多次中断，已停止重试',
  );
  assert.equal(
    jobErrorText('stylize failed, model m gets no stylized GLB: timed out after 1200 s'),
    '风格化超时，这个模型没有风格化版本',
  );
  assert.equal(
    jobErrorText('stylize failed, model m gets no stylized GLB: ValueError: no outline'),
    '风格化模型生成失败，这个模型没有风格化版本',
  );
  for (const other of ['COLMAP failed: boom', 'ValueError: photo x has no processed copy', '', null]) {
    assert.equal(jobErrorText(other), '重建程序出错');
  }
});

test('a HEIC/HEIF file the server cannot decode gets a format hint instead of "unreadable"', () => {
  const result = (filename, reasons) => ({ filename, reasons, accepted: false, warnings: [] });
  for (const name of ['IMG_0001.HEIC', 'photo.heif', 'a.b.heic']) {
    assert.match(uploadReasonText(result(name, ['unreadable_image'])), /暂不支持 HEIC 格式/, name);
  }
  assert.equal(uploadReasonText(result('broken.jpg', ['unreadable_image'])), '无法读取图片');
  assert.equal(uploadReasonText(result('x.heic', ['no_gps', 'too_far'])), '缺少GPS定位、距离地标太远');
});

test('the API client has no way to fetch exact camera poses (D30: only the rounded public poses are served)', async () => {
  const api = await import('../api.js');
  assert.deepEqual(Object.keys(api).filter((k) => /camera/i.test(k)), []);
});
