// Run from the repo root: node --test 'web/tests/*.test.mjs'
import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

import {
  ApiError,
  JOB_KIND_TEXT,
  JOB_STATUS_TEXT,
  MODEL_KIND_TEXT,
  REASON_TEXT,
  TERMS_VERSION,
  WARNING_TEXT,
  acceptTerms,
  codeListText,
  codeText,
  createLandmark,
  errorMessage,
  getLandmark,
  getPoses,
  isActiveJob,
  jobErrorText,
  listLandmarks,
  ownPhotos,
  rebuild,
  refusalType,
  register,
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
    server_error: '服务器没能保存这个文件，其他照片已保留，请稍后重试', // crowd3d/api.py R_SERVER_ERROR
    no_heading: '缺少指南针朝向',
    magnetic_heading: '使用磁北朝向',
    subject_unchecked: '没有检查建筑是不是照片主体',
    heading_away_unverified: '指南针朝向背对建筑，但没有标明是真北，照片已保留', // crowd3d/validation.py W_HEADING_AWAY
  };
  for (const [code, text] of Object.entries(expected)) assert.equal(codeText(code), text, code);
  assert.equal(Object.keys(REASON_TEXT).length + Object.keys(WARNING_TEXT).length, Object.keys(expected).length);
  assert.equal(codeText('something_new'), 'something_new');
  assert.equal(codeText('toString'), 'toString', 'an Object.prototype name is no code');
  assert.equal(codeListText(['no_gps', 'too_far', 'no_heading']), '缺少GPS定位、距离地标太远、缺少指南针朝向');
  assert.equal(codeListText([]), '');
});

test('the codes in English (D32): reasons, warnings, job states and kinds, model kinds', () => {
  inLang('en', () => {
    assert.equal(codeText('no_gps'), 'No GPS location');
    assert.equal(
      codeText('not_subject'),
      'More than 50 m from the building, and the building is not the main subject: move closer, or fill the frame with the building',
    );
    assert.equal(codeText('server_error'), 'The server could not store this file (the other photos are kept): try again later');
    assert.equal(codeText('magnetic_heading'), 'Heading relative to magnetic north');
    assert.equal(codeText('something_new'), 'something_new');
    assert.equal(codeListText(['no_gps', 'too_far']), 'No GPS location. Too far from the landmark');
    assert.equal(JOB_STATUS_TEXT.running, 'Running');
    assert.equal(JOB_STATUS_TEXT.failed, 'Failed');
    assert.equal(JOB_KIND_TEXT.stylize, 'stylization');
    assert.deepEqual({ ...MODEL_KIND_TEXT }, { face: 'Facade model', full: 'Full model' });
  });
  // the same tables in Chinese, unchanged
  assert.deepEqual({ ...JOB_STATUS_TEXT }, { queued: '排队中', running: '生成中', done: '已完成', failed: '失败' });
  assert.deepEqual({ ...JOB_KIND_TEXT }, { stylize: '风格化' });
  assert.deepEqual({ ...MODEL_KIND_TEXT }, { face: '立面模型', full: '完整模型' });
  assert.equal(JOB_STATUS_TEXT.nothing, undefined);
});

// FastAPI's own 422 items, as crowd3d/api.py sends them (msg and input are the server's, never shown)
const tooBig = { type: 'less_than_equal', loc: ['body', 'lat'], msg: 'Input should be less than or equal to 90', input: 91, ctx: { le: 90 } };
const noName = { type: 'missing', loc: ['body', 'name'], msg: 'Field required', input: {} };

test("errorMessage: a text per status; the server's own detail is never shown, ApiError.detail keeps it", () => {
  assert.equal(errorMessage(0, null), '无法连接服务器，请检查网络后重试');
  assert.equal(errorMessage(403, { detail: 'wrong manage token' }), '没有权限进行这项操作');
  assert.equal(errorMessage(404, { detail: "unknown landmark 'x'" }), '找不到要访问的内容，可能已被删除');
  assert.equal(errorMessage(409, { detail: 'landmark exists' }), '和服务器上已有的内容冲突');
  assert.equal(errorMessage(413, { detail: 'request body over 1049576 bytes' }), '上传的内容太大，请少选几张照片后重试');
  assert.equal(errorMessage(500, null), '服务器出错了，请稍后重试（HTTP 500）');
  assert.equal(errorMessage(503, { detail: 'x' }), '服务器出错了，请稍后重试（HTTP 503）');
  assert.equal(errorMessage(400, { detail: 'There was an error parsing the body' }), '请求失败（HTTP 400）');
  assert.equal(errorMessage(422, { detail: 'not a list' }), '请求失败（HTTP 422）');
  assert.equal(errorMessage(422, { detail: [] }), '请求失败（HTTP 422）');
  assert.equal(errorMessage(1000, null), '请求失败（HTTP 1000）', 'the status as it is, no grouping');
  inLang('en', () => {
    assert.equal(errorMessage(0, null), 'Could not reach the server. Check your connection and try again.');
    assert.equal(errorMessage(403, null), "You don't have permission to do this.");
    assert.equal(errorMessage(404, null), 'Not found. It may have been removed.');
    assert.equal(errorMessage(409, null), 'This conflicts with something already on the server.');
    assert.equal(errorMessage(413, { detail: 'request body over 1049576 bytes' }), 'The upload is too large. Select fewer photos and try again.');
    assert.equal(errorMessage(502, null), 'Something went wrong on the server. Try again later (HTTP 502).');
    assert.equal(errorMessage(400, null), 'The request failed (HTTP 400).');
  });
  const err = new ApiError(409, errorMessage(409, { detail: 'landmark exists' }), { detail: 'landmark exists' });
  assert.equal(err.detail, 'landmark exists');
  assert.equal(new ApiError(0, 'x').detail, null);
});

test('errorMessage: each 422 item as its field and what is wrong with it, in the page language', () => {
  assert.equal(errorMessage(422, { detail: [tooBig, noName] }), '纬度：不能大于 90；名称：必填');
  const item = (type, field, ctx) => ({ detail: [{ type, loc: ['body', field], msg: 'technical', input: null, ctx }] });
  assert.equal(errorMessage(422, item('greater_than_equal', 'lon', { ge: -180 })), '经度：不能小于 -180');
  assert.equal(errorMessage(422, item('greater_than', 'radius_m', { gt: 0 })), '半径：必须大于 0');
  assert.equal(errorMessage(422, item('less_than_equal', 'radius_m', { le: 5000 })), '半径：不能大于 5,000');
  assert.equal(errorMessage(422, item('string_too_long', 'name', { max_length: 100 })), '名称：最多 100 个字符');
  assert.equal(errorMessage(422, item('string_too_short', 'name', { min_length: 1 })), '名称：至少 1 个字符');
  assert.equal(errorMessage(422, item('string_pattern_mismatch', 'id', { pattern: '^[a-z]' })), '标识：取值无效');
  assert.equal(errorMessage(422, item('value_error', 'nickname')), '昵称：取值无效'); // registering (D33)
  assert.equal(errorMessage(422, item('toString', 'files')), '照片：取值无效', 'an Object.prototype name is no type');
  assert.equal(errorMessage(422, item('less_than_equal', 'lat', {})), '纬度：取值无效', 'no limit in ctx');
  assert.equal(errorMessage(422, item('missing', 'files')), '照片：必填');
  // a list item's index after the field; a field the client does not know shows as the server names it
  assert.equal(errorMessage(422, { detail: [{ type: 'missing', loc: ['body', 'files', 0] }] }), '照片：必填');
  assert.equal(errorMessage(422, item('literal_error', 'share')), 'share：取值无效');
  assert.equal(errorMessage(422, item('missing', 'constructor')), 'constructor：必填');
  // nothing to name: the whole body
  assert.equal(errorMessage(422, { detail: [{ type: 'missing', loc: ['body'] }, null] }), '提交的内容有误；提交的内容有误');
  inLang('en', () => {
    assert.equal(errorMessage(422, { detail: [tooBig, noName] }), 'Latitude: must be at most 90; Name: required');
    assert.equal(errorMessage(422, item('less_than_equal', 'radius_m', { le: 5000 })), 'Radius: must be at most 5,000');
    assert.equal(errorMessage(422, item('string_too_long', 'name', { max_length: 100 })), 'Name: at most 100 characters');
    assert.equal(errorMessage(422, item('string_too_short', 'name', { min_length: 1 })), 'Name: at least 1 character');
    assert.equal(errorMessage(422, item('value_error', 'nickname')), 'Nickname: not a valid value');
    assert.equal(errorMessage(422, { detail: [{ type: 'missing', loc: ['body'] }] }), 'Something in the request is not valid');
  });
});

test('errorMessage shows its own text for an upload without the current user agreement (D32)', () => {
  const detail = (type) => ({
    detail: [{ type, loc: ['body', 'terms'], msg: 'uploading needs agreeing to the user agreement terms-v1 (/terms.html)',
      input: null, ctx: { version: 'terms-v1', url: '/terms.html' } }],
  });
  assert.equal(errorMessage(422, detail('terms_required')), '请先阅读并同意《用户协议》');
  assert.equal(errorMessage(422, detail('terms_outdated')), '用户协议已更新，请刷新页面后重新阅读并同意');
  inLang('en', () => {
    assert.equal(errorMessage(422, detail('terms_required')), 'Please read and accept the Terms of Use first.');
    assert.equal(errorMessage(422, detail('terms_outdated')), 'The Terms of Use have changed. Reload the page and accept the new version.');
  });
  // other types on that field: its label and a text by type
  assert.equal(errorMessage(422, { detail: [{ type: 'missing', loc: ['body', 'terms'], msg: 'Field required' }] }), '用户协议：必填');
});

test('errorMessage and refusalType: a request without a registration the server knows (D33)', () => {
  // crowd3d/api.py _register_required: at loc contributor_token, input null (the token never comes back)
  const body = {
    detail: [{ type: 'register_required', loc: ['body', 'contributor_token'], msg: 'uploading needs a registered contributor (POST /api/contributors)', input: null }],
  };
  assert.equal(errorMessage(422, body), '请先登记昵称并同意《用户协议》');
  inLang('en', () => assert.equal(errorMessage(422, body), 'Please register a nickname and accept the Terms of Use first.'));
  // an upload of a contributor who has not accepted the current version: terms_outdated at the token's loc
  const outdated = { detail: [{ type: 'terms_outdated', loc: ['body', 'contributor_token'], msg: 'x', input: null, ctx: { version: 'terms-v1' } }] };
  assert.equal(errorMessage(422, outdated), '用户协议已更新，请刷新页面后重新阅读并同意');

  assert.equal(refusalType(new ApiError(422, 'x', body)), 'register_required');
  assert.equal(refusalType(new ApiError(422, 'x', outdated)), 'terms_outdated');
  assert.equal(refusalType(new ApiError(422, 'x', { detail: [{ loc: ['body'] }] })), null, 'no type');
  assert.equal(refusalType(new ApiError(422, 'x', { detail: 'not a list' })), null);
  assert.equal(refusalType(new ApiError(422, 'x', null)), null);
  assert.equal(refusalType(new ApiError(403, 'x', body)), null, 'only a 422 is a refusal by type');
  assert.equal(refusalType(new TypeError('Failed to fetch')), null);
  assert.equal(refusalType(null), null);
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
    assert.equal(err.message, '和服务器上已有的内容冲突');
    assert.equal(err.detail, 'duplicate id'); // the server's technical text, kept
    return true;
  });
});

test('non-JSON error bodies and network failures become readable ApiErrors', async () => {
  mockFetch(502, '<html>bad gateway</html>', { raw: true });
  await assert.rejects(
    listLandmarks(),
    (err) => err.status === 502 && err.message === '服务器出错了，请稍后重试（HTTP 502）' && err.body === null && err.detail === null,
  );

  globalThis.fetch = async () => {
    throw new TypeError('Failed to fetch');
  };
  await assert.rejects(
    listLandmarks(),
    (err) => err instanceof ApiError && err.status === 0 && err.message === '无法连接服务器，请检查网络后重试',
  );
  await inLang('en', () =>
    assert.rejects(listLandmarks(), (err) => err.message === 'Could not reach the server. Check your connection and try again.'),
  );
});

test('register and acceptTerms post JSON with the agreement version; the token only in the body (D33)', async () => {
  const token = 'a1'.repeat(32);
  mockFetch(201, { nickname: '小明', terms: 'terms-v1', contributor_token: token });
  assert.deepEqual(await register({ nickname: '小明' }), { nickname: '小明', terms: 'terms-v1', contributor_token: token });
  assert.equal(calls[0].url, '/api/contributors');
  assert.equal(calls[0].init.method, 'POST');
  assert.equal(calls[0].init.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(calls[0].init.body), { nickname: '小明', terms: 'terms-v1' });

  mockFetch(200, { nickname: '小明', terms: 'terms-v1' });
  assert.deepEqual(await acceptTerms(token), { nickname: '小明', terms: 'terms-v1' });
  assert.equal(calls[0].url, '/api/contributors/accept');
  assert.equal(calls[0].init.method, 'POST');
  assert.deepEqual(JSON.parse(calls[0].init.body), { contributor_token: token, terms: TERMS_VERSION });
  assert.ok(!calls[0].url.includes(token), 'never in the URL');
  assert.ok(!JSON.stringify(calls[0].init.headers).includes(token), 'never in a header');

  // a refusal by type: the page acts on it (refusalType), its text in the page language
  mockFetch(422, { detail: [{ type: 'register_required', loc: ['body', 'contributor_token'], msg: 'x', input: null }] });
  await assert.rejects(acceptTerms(token), (err) => {
    assert.equal(refusalType(err), 'register_required');
    assert.equal(err.message, '请先登记昵称并同意《用户协议》');
    return true;
  });
  mockFetch(422, { detail: [{ type: 'value_error', loc: ['body', 'nickname'], msg: 'x', input: '' }] });
  await assert.rejects(register({ nickname: ' ' }), (err) => err.message === '昵称：取值无效' && refusalType(err) === 'value_error');
});

test('uploadPhotos sends multipart with field names from the contract: the contributor token, never a nickname or the agreement (D33)', async () => {
  const token = 'c3'.repeat(32);
  mockFetch(200, { results: [], triggered_jobs: [] });
  const files = [new File(['a'], 'one.jpg', { type: 'image/jpeg' }), new File(['b'], 'two.jpg', { type: 'image/jpeg' })];
  await uploadPhotos({ files, token, landmarkId: 'demo-building' });
  const form = calls[0].init.body;
  assert.equal(calls[0].url, '/api/photos');
  assert.equal(calls[0].init.method, 'POST');
  assert.ok(form instanceof FormData);
  assert.deepEqual(
    form.getAll('files').map((f) => f.name),
    ['one.jpg', 'two.jpg'],
  );
  assert.deepEqual(form.getAll('contributor_token'), [token]);
  assert.equal(form.get('landmark_id'), 'demo-building');
  assert.deepEqual([...new Set(form.keys())].sort(), ['contributor_token', 'files', 'landmark_id']);
  // D34: the text fields before the files, in the order the parts are sent. The server stops reading the form at the
  // 51st file; a token after the files would then be unread and the answer register_required, which makes the page
  // drop the stored registration.
  assert.deepEqual([...form.keys()], ['contributor_token', 'landmark_id', 'files', 'files']);
  assert.equal(calls[0].init.headers['Content-Type'], undefined, 'browser must set the multipart boundary');
  // the token never in the URL or a header (they end up in logs)
  assert.ok(!calls[0].url.includes(token));
  assert.ok(!JSON.stringify(calls[0].init.headers).includes(token));
  assert.equal(TERMS_VERSION, 'terms-v1');

  mockFetch(200, { results: [], triggered_jobs: [] });
  await uploadPhotos({ files, token });
  assert.equal(calls[0].init.body.has('landmark_id'), false);
  assert.deepEqual([...calls[0].init.body.keys()], ['contributor_token', 'files', 'files']);
  // 51 files (the page stops at 50 itself; another caller may not): the token is still the first part
  mockFetch(200, { results: [], triggered_jobs: [] });
  const many = Array.from({ length: 51 }, (_, i) => new File(['x'], `${i}.jpg`, { type: 'image/jpeg' }));
  await uploadPhotos({ files: many, token, landmarkId: 'demo-building' });
  assert.deepEqual([...calls[0].init.body.keys()].slice(0, 3), ['contributor_token', 'landmark_id', 'files']);
  assert.equal(calls[0].init.body.getAll('files').length, 51);
  // D33: the server takes the nickname and the agreement from the registration; never the D30 share_location field
  for (const extra of [{ userId: 'u' }, { user_id: 'u' }, { terms: 'terms-v1' }, { shareLocation: true }]) {
    mockFetch(200, { results: [], triggered_jobs: [] });
    await uploadPhotos({ files, token, ...extra });
    const body = calls[0].init.body;
    for (const field of ['user_id', 'terms', 'share_location']) assert.equal(body.has(field), false, `${field} ${JSON.stringify(extra)}`);
  }
  // no token: none sent (the server answers register_required)
  mockFetch(200, { results: [], triggered_jobs: [] });
  await uploadPhotos({ files, token: null });
  assert.equal(calls[0].init.body.has('contributor_token'), false);
});

test('withdrawShareLocation posts share false with the manage token (D30)', async () => {
  mockFetch(200, { photo_id: 'p/1', share_location: false });
  const token = 'a'.repeat(64);
  assert.deepEqual(await withdrawShareLocation('p/1', token), { photo_id: 'p/1', share_location: false });
  assert.equal(calls[0].url, '/api/photos/p%2F1/share_location');
  assert.equal(calls[0].init.method, 'POST');
  assert.deepEqual(JSON.parse(calls[0].init.body), { share: false, token });

  mockFetch(403, { detail: 'wrong token for this photo' });
  await assert.rejects(withdrawShareLocation('p', token), (err) => err instanceof ApiError && err.status === 403);
});

test("ownPhotos posts the contributor token (and the landmark) as JSON: a registration's own shared photos, the token only in the body (D34)", async () => {
  const token = 'b2'.repeat(32);
  const photos = [{ photo_id: 'p1', landmark_id: 'lm 1', public_id: '0123456789ab', manage_token: 'c'.repeat(64) }];
  mockFetch(200, { photos });
  assert.deepEqual(await ownPhotos(token, 'lm 1'), { photos });
  assert.equal(calls[0].url, '/api/contributors/photos');
  assert.equal(calls[0].init.method, 'POST');
  assert.equal(calls[0].init.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(calls[0].init.body), { contributor_token: token, landmark_id: 'lm 1' });
  assert.ok(!calls[0].url.includes(token), 'never in the URL');
  assert.ok(!JSON.stringify(calls[0].init.headers).includes(token), 'never in a header');
  // every landmark: no landmark field
  mockFetch(200, { photos: [] });
  assert.deepEqual(await ownPhotos(token), { photos: [] });
  assert.deepEqual(JSON.parse(calls[0].init.body), { contributor_token: token });
  // a registration the server does not know: the typed refusal, like the other contributor routes
  mockFetch(422, { detail: [{ type: 'register_required', loc: ['body', 'contributor_token'], msg: 'x', input: null }] });
  await assert.rejects(ownPhotos(token, 'lm'), (err) => refusalType(err) === 'register_required' && err.message === '请先登记昵称并同意《用户协议》');
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
    // a no-break space before AM/PM (i18n.js fmtDateTime)
    assert.equal(fmtTime(oct21, later), 'Oct 21, 9:05\u00a0AM');
  });
  assert.equal(fmtCount(1234), '1,234');
  assert.equal(fmtTime(1000, 1000 + 60), '1 分钟前');
  assert.equal(fmtTime(oct21, later), '10月21日 09:05');
  assert.equal(fmtTime(new Date(2026, 0, 3, 0, 7).getTime() / 1000, later), '1月3日 00:07');
});

test('job errors from the reconstructor get their text (Chinese here; English below)', () => {
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
  assert.equal(jobErrorText('only 1234 of 5678 photos could be registered'), '5678 张照片里只有 1234 张能互相对上', 'counts as written');
  inLang('en', () => {
    assert.equal(
      jobErrorText('only 2 of 9 photos could be registered; need at least 3 overlapping views'),
      'Only 2 of 9 photos could be matched with each other',
    );
    assert.equal(
      jobErrorText('stylize failed, model m gets no stylized GLB: timed out after 1200 s'),
      'Stylizing timed out, so this model has no stylized version',
    );
    assert.equal(jobErrorText('need at least 2 photos'), 'Too few usable photos');
    assert.equal(jobErrorText('COLMAP failed: boom'), 'The reconstruction software ran into an error');
  });
});

test('a HEIC/HEIF file the server cannot decode gets a format hint instead of "unreadable"', () => {
  const result = (filename, reasons) => ({ filename, reasons, accepted: false, warnings: [] });
  for (const name of ['IMG_0001.HEIC', 'photo.heif', 'a.b.heic']) {
    assert.match(uploadReasonText(result(name, ['unreadable_image'])), /暂不支持 HEIC 格式/, name);
  }
  assert.equal(uploadReasonText(result('broken.jpg', ['unreadable_image'])), '无法读取图片');
  assert.equal(uploadReasonText(result('x.heic', ['no_gps', 'too_far'])), '缺少GPS定位、距离地标太远');
  assert.equal(uploadReasonText(result('x.jpg', ['server_error'])), '服务器没能保存这个文件，其他照片已保留，请稍后重试');
  inLang('en', () => {
    assert.equal(
      uploadReasonText(result('IMG_0001.HEIC', ['unreadable_image', 'no_gps'])),
      'HEIC photos are not supported yet: export them as JPEG, or set the camera format to "Most Compatible". No GPS location',
    );
    assert.equal(uploadReasonText(result('broken.jpg', ['unreadable_image'])), 'The image could not be read');
  });
});

test('the API client has no way to fetch exact camera poses (D30: only the rounded public poses are served)', async () => {
  const api = await import('../api.js');
  assert.deepEqual(Object.keys(api).filter((k) => /camera/i.test(k)), []);
});
