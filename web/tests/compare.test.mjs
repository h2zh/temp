// Run from the repo root: node --test 'web/tests/*.test.mjs'
// "View from this photo" as a comparison of the photo with the clay model seen from its pose (web/compare.js, D32):
// the layout, the alignment of the model pane with the photo, the photo's URL, the texts in both languages, the URL
// and stored choices, and the DOM on the fake document. The 3-D side (the stage resized to the model pane, the camera
// at the pose) is web/photoposes.js's and is checked in a browser.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as C from '../compare.js';
import { missing } from '../i18n.js';
import * as P from '../photoposes.js';
import { inLang, installDom, text } from './fake-dom.mjs';

// rows as crowd3d/photoposes.py public_row writes them (whole metres and degrees, hfov to whole degrees too)
const SFM = { id: '0a0a0a0a0a0a', src: 'sfm', p: [84, -4, 2], yaw: 265, pitch: 8, roll: 0, hfov: 40, aspect: 1.33, sp: 16, sy: 21 };
const SNAPPED = { id: '1b1b1b1b1b1b', src: 'sfm', p: [23, -1, 2], yaw: 277, pitch: 11, roll: 0, hfov: 65, aspect: 1.33, sp: 7, sy: 7 };
const EXIF = { id: '2c2c2c2c2c2c', src: 'exif', p: [107, -39, null], yaw: 269, pitch: null, roll: null, hfov: 60, aspect: 1.33, sp: 41, sy: 31 };
const PORTRAIT = { id: '3d3d3d3d3d3d', src: 'loc', p: [-30, 12, 1], yaw: 95, pitch: -3, roll: 2, hfov: 45, aspect: 0.71, sp: 9, sy: 6 };
const ROUNDING = { position_m: 1, angle_deg: 1, fov_deg: 1 };
const POSES = '/api/landmarks/florence-baptistery/poses';
const STORE_ID = 'e4f1c6eca80c48ce83b7b3d9c42406e7';
// Chinese characters and CJK / full-width punctuation
const CJK = /[\p{Script=Han}　-〿＀-￯]/u;

const BARS = { top: 70, bottom: 60 };
const VIEWPORTS = [[1280, 860, 'row'], [390, 844, 'column']];
const area = (r) => r.w * r.h;
const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

test('layout: side by side on a wide screen, stacked on a phone, for a landscape and a portrait photo', () => {
  for (const [vw, vh, dir] of VIEWPORTS) {
    for (const aspect of [1.33, 0.71]) {
      const L = C.compareLayout({ vw, vh, aspect, mode: 'pair', ...BARS });
      const what = `${vw}x${vh} at ${aspect}`;
      assert.equal(L.dir, dir, what);
      for (const r of [L.photo, L.model]) for (const v of Object.values(r)) assert.ok(Number.isInteger(v), `${what}: whole pixels`);
      // two equal panes of the photo's aspect, apart, inside the space between the bars
      assert.deepEqual([L.photo.w, L.photo.h], [L.model.w, L.model.h], what);
      assert.ok(Math.abs(L.model.w / L.model.h - aspect) <= 1 / L.model.h, `${what}: ${L.model.w}x${L.model.h}`);
      assert.ok(!overlaps(L.photo, L.model), what);
      for (const r of [L.photo, L.model]) {
        assert.ok(r.x >= 12 && r.x + r.w <= vw - 12, `${what}: inside horizontally`);
        assert.ok(r.y >= BARS.top + 12 && r.y + r.h <= vh - BARS.bottom - 12, `${what}: inside vertically`);
      }
      // the photo left of or above the model, 10 px apart
      if (dir === 'row') assert.deepEqual([L.model.x - L.photo.x - L.photo.w, L.model.y], [10, L.photo.y], what);
      else assert.deepEqual([L.model.y - L.photo.y - L.photo.h, L.model.x], [10, L.photo.x], what);
      // the larger panes of the two arrangements
      const other = C.compareLayout({ vw: vh, vh: vw, aspect, mode: 'pair', ...BARS });
      assert.ok(area(L.model) > 0 && other.model.w > 0);
      // overlay and photo-only: one pane, the larger one, the same rectangle for the photo and the model
      for (const mode of ['overlay', 'photo']) {
        const one = C.compareLayout({ vw, vh, aspect, mode, ...BARS });
        assert.equal(one.photo, one.model, `${what} ${mode}`);
        assert.equal(one.dir, dir, `${what} ${mode}: the pair's direction for its button`);
        assert.ok(area(one.model) >= area(L.model), `${what} ${mode}`);
        assert.ok(one.model.x >= 12 && one.model.x + one.model.w <= vw - 12 && one.model.y >= BARS.top + 12 && one.model.y + one.model.h <= vh - BARS.bottom - 12);
        // centred
        assert.ok(Math.abs(one.model.x + one.model.w / 2 - vw / 2) <= 1, `${what} ${mode}: centred`);
      }
      // the group is centred too
      const gx0 = Math.min(L.photo.x, L.model.x);
      const gx1 = Math.max(L.photo.x + L.photo.w, L.model.x + L.model.w);
      assert.ok(Math.abs((gx0 + gx1) / 2 - vw / 2) <= 1, `${what}: centred`);
    }
  }
  // the numbers on the two screens of the browser check
  assert.deepEqual(C.compareLayout({ vw: 1280, vh: 860, aspect: 1.33, mode: 'pair', ...BARS }).model, { x: 645, y: 201, w: 623, h: 468 });
  assert.deepEqual(C.compareLayout({ vw: 390, vh: 844, aspect: 0.71, mode: 'pair', ...BARS }).photo, { x: 74, y: 82, w: 242, h: 340 });
  // a tie goes side by side; taller bars shrink the panes; a missing aspect lays out a 4:3 placeholder
  assert.equal(C.compareLayout({ vw: 524, vh: 524, aspect: 1, mode: 'pair' }).dir, 'row');
  const tall = C.compareLayout({ vw: 1280, vh: 860, aspect: 1.33, mode: 'pair', top: 300, bottom: 200 });
  assert.ok(tall.model.h < 468 && tall.model.y >= 312 && tall.model.y + tall.model.h <= 860 - 212);
  assert.deepEqual(C.compareLayout({ vw: 800, vh: 600, aspect: null, mode: 'photo' }).model,
    C.compareLayout({ vw: 800, vh: 600, aspect: 4 / 3, mode: 'photo' }).model);
  // fractional bar edges are rounded outwards
  const frac = C.compareLayout({ vw: 1280, vh: 860, aspect: 1.33, mode: 'overlay', top: 70.4, bottom: 59.2 });
  assert.ok(frac.model.y >= 71 + 12 && frac.model.y + frac.model.h <= 860 - 60 - 12);
});

test('alignment: the model pane is the photo\'s frame; the camera takes the photo\'s own field of view', () => {
  let worst = { frame: 0, fov: 0 };
  for (const [vw, vh] of [...VIEWPORTS, [1920, 1080], [768, 1024], [844, 390]]) {
    for (const aspect of [1.33, 0.71, 1.5, 1.78, 0.56, 1]) {
      for (const hfov of [40, 65, 90]) {
        for (const mode of ['pair', 'overlay', 'photo']) {
          const row = { ...SFM, aspect, hfov };
          const { model } = C.compareLayout({ vw, vh, aspect, mode, ...BARS });
          const vp = P.viewPose(row, model.w, model.h);
          const fr = { x: vp.frame.x * model.w, y: vp.frame.y * model.h, w: vp.frame.w * model.w, h: vp.frame.h * model.h };
          const what = `${vw}x${vh} ${aspect} ${hfov}° ${mode}`;
          for (const d of [fr.x, fr.y, fr.w - model.w, fr.h - model.h]) {
            assert.ok(Math.abs(d) <= 1, `${what}: the photo frame within 1 px of the pane`);
            worst.frame = Math.max(worst.frame, Math.abs(d));
          }
          const dfov = Math.abs(vp.fov - P.vfovOf(hfov, aspect));
          assert.ok(dfov <= 0.05, `${what}: fov ${vp.fov} vs ${P.vfovOf(hfov, aspect)}`);
          worst.fov = Math.max(worst.fov, dfov);
        }
      }
    }
  }
  assert.ok(worst.frame < 1 && worst.fov < 0.05, JSON.stringify(worst));
});

test('the photo URL: the poses path, then the 12-hex public id; nothing else ever', () => {
  assert.equal(C.photoUrl(POSES, SFM.id), `${POSES}/${SFM.id}/photo`);
  assert.equal(C.photoUrl(`${POSES}?as_of=1&x=y#z`, SFM.id), `${POSES}/${SFM.id}/photo`);
  assert.equal(C.photoUrl(`${POSES}/`, SFM.id), `${POSES}/${SFM.id}/photo`);
  assert.equal(C.photoUrl('http://localhost/api/landmarks/lm/poses', SFM.id), `/api/landmarks/lm/poses/${SFM.id}/photo`);
  for (const id of ['../x', STORE_ID, SFM.id.toUpperCase(), `${SFM.id}0`, SFM.id.slice(1), '', null, undefined, 12, `${SFM.id}/../x`]) {
    assert.equal(C.photoUrl(POSES, id), null, `id ${id}`);
  }
  // a poses path parsePoseParams would not take
  for (const path of ['https://evil.example/api/landmarks/lm/poses', '//evil.example/poses', 'javascript:alert(1)', '', null]) {
    assert.equal(C.photoUrl(path, SFM.id), null, `path ${path}`);
  }
  assert.match(C.photoUrl(POSES, SFM.id), /^\/api\/landmarks\/florence-baptistery\/poses\/[0-9a-f]{12}\/photo$/);
  assert.equal(C.PUBLIC_ID_RE.test(STORE_ID), false);
});

test('rows without a measured pose or field of view: no model view, the photo alone with its honest label (D30)', () => {
  const blocked = [EXIF, { ...EXIF, p: [1, 2, 3], pitch: 0, roll: 0 }, { ...SFM, hfov: null }, { ...SFM, aspect: null },
    { ...SFM, pitch: null }, { ...SFM, roll: null }, { ...SFM, p: [84, -4, null] }];
  for (const row of blocked) {
    assert.equal(P.viewPose(row, 800, 600), null);
    assert.deepEqual(C.cardActions(row), [{ key: 'photo', note: P.viewFromBlocked(row) }]);
    assert.equal(C.compareBanner(row, ROUNDING).warn, null);
  }
  for (const row of [SFM, SNAPPED, PORTRAIT]) assert.deepEqual(C.cardActions(row), [{ key: 'view' }]);
  // Chinese
  assert.deepEqual(C.compareBanner(EXIF, ROUNDING), {
    title: '只有照片（不能和素模对照）',
    detail: '来源：手机 GPS + 罗盘（未经模型核对）；位置 ±41 米，朝向 ±31°（2σ）。手机 GPS + 罗盘只给出大致的位置和朝向，没有测出拍摄高度和仰角，无法还原拍摄角度，不能和素模对照。',
    warn: null,
  });
  assert.equal(C.noCompareText('exif'), '不能和素模对照：手机 GPS + 罗盘只给出大致的位置和朝向，没有测出拍摄高度和仰角，无法还原拍摄角度。');
  assert.equal(C.noCompareText('fov'), '不能和素模对照：没有测出视角（焦距或画幅未知），无法还原拍摄角度。');
  assert.match(C.compareBanner({ ...SFM, pitch: null }, ROUNDING).detail, /^来源：模型重建；位置 ±16 米，朝向 ±21°（2σ）。没有测出拍摄高度、仰角或侧倾，无法还原拍摄角度，不能和素模对照。$/);
  // English: the reason starts the sentence
  inLang('en', () => {
    assert.deepEqual(C.compareBanner(EXIF, ROUNDING), {
      title: "Photo only (can't be compared with the model)",
      detail: "Source: Phone GPS and compass (not checked against the model). Position ±41 m, heading ±31° (2σ). The phone's GPS and compass give only a rough position and heading; the camera's height and tilt weren't measured, so it can't be compared with the model.",
      warn: null,
    });
    assert.equal(C.compareBanner({ ...SFM, aspect: null }, ROUNDING).detail,
      "Source: 3D reconstruction. Position ±16 m, heading ±21° (2σ). The field of view wasn't measured (unknown focal length or frame size), so it can't be compared with the model.");
    assert.equal(C.noCompareText('exif'), "Can't be compared with the model: the phone's GPS and compass give only a rough position and heading; the camera's height and tilt weren't measured.");
    assert.equal(C.noCompareText('pose'), "Can't be compared with the model: the camera's height, tilt or roll wasn't measured.");
  });
});

test('the top bar: source, 2-sigma ranges, the rounding with the field-of-view step; the warning only beyond 10 m', () => {
  assert.deepEqual(C.compareBanner(SFM, ROUNDING), {
    title: '照片和从同一位置看到的素模',
    detail: '来源：模型重建；位置 ±16 米，朝向 ±21°（2σ）；位置取整到 1 米、朝向取整到 1°、视角取整到 1°。素模只有轮廓和高度，没有照片里的细节。',
    warn: '相机可能在这个范围里的任何地方，素模和照片可能对不上。',
  });
  assert.equal(C.compareBanner(SNAPPED, ROUNDING).warn, null);
  assert.equal(C.compareBanner({ ...SNAPPED, sp: 10 }, ROUNDING).warn, null);
  assert.notEqual(C.compareBanner({ ...SNAPPED, sp: 11 }, ROUNDING).warn, null);
  assert.match(C.compareBanner(SNAPPED, { position_m: 2, angle_deg: 3, fov_deg: 10 }).detail, /位置取整到 2 米、朝向取整到 3°、视角取整到 10°/);
  assert.match(C.compareBanner(SNAPPED).detail, /视角取整到 1°/);
  inLang('en', () => {
    assert.deepEqual(C.compareBanner(SFM, ROUNDING), {
      title: 'The photo and the clay model seen from the same spot',
      detail: "Source: 3D reconstruction. Position ±16 m, heading ±21° (2σ); position rounded to 1 m, heading to 1°, field of view to 1°. The clay model shows only the outline and heights, none of the photo's detail.",
      warn: 'The camera could be anywhere in this range, so the model may not line up with the photo.',
    });
    assert.match(C.compareBanner(PORTRAIT, ROUNDING).detail, /^Source: Matched to the model\. Position ±9 m, heading ±6° \(2σ\);/);
    assert.equal(C.compareBanner(PORTRAIT, ROUNDING).warn, null);
    for (const row of [SFM, SNAPPED, PORTRAIT, EXIF]) for (const v of Object.values(C.compareBanner(row, ROUNDING))) assert.ok(v === null || !CJK.test(v), v);
  });
  // nothing about who took it, when, or which stored photo it is, whatever the row carries
  for (const l of ['zh', 'en']) {
    inLang(l, () => {
      for (const row of [SFM, EXIF]) {
        const extra = { ...row, t: '2019-07-14T10:00:00', user_id: 'alice', store_id: STORE_ID, filename: 'IMG_0001.jpg' };
        const all = Object.values(C.compareBanner(extra, ROUNDING)).join('\n');
        assert.doesNotMatch(all, /alice|2019|IMG_0001|[0-9a-f]{12}|uploaded by|拍摄者|上传者/i, l);
        assert.deepEqual(C.compareBanner(extra, ROUNDING), C.compareBanner(row, ROUNDING));
      }
    });
  }
});

test('URL and stored choices: cmp and opacity parsed and clamped, the URL over the stored choice over the defaults', () => {
  const o = (q) => P.parsePoseParams(q, 'https://c.example');
  assert.deepEqual([o('').cmp, o('').opacity], [null, null]);
  assert.deepEqual([o('?cmp=overlay&opacity=30').cmp, o('?cmp=overlay&opacity=30').opacity], ['overlay', 30]);
  assert.equal(o('?cmp=pair').cmp, 'pair');
  for (const bad of ['photo', 'Overlay', 'side', '']) assert.equal(o(`?cmp=${bad}`).cmp, null, bad);
  const cases = { 150: 100, 100: 100, '-5': 0, 0: 0, 42: 40, 43: 45, 52.5: 55, ' 70 ': 70 };
  for (const [v, want] of Object.entries(cases)) assert.equal(o(`?opacity=${encodeURIComponent(v)}`).opacity, want, `opacity=${v}`);
  for (const bad of ['abc', '', '1e2', '50%', 'NaN', 'Infinity', '0x20']) assert.equal(o(`?opacity=${encodeURIComponent(bad)}`).opacity, null, `opacity=${bad}`);
  // the stored choice (localStorage 'crowd3d.compare'): read defensively
  const store = (raw) => ({ getItem: (k) => (k === C.PREFS_KEY ? raw : null) });
  assert.deepEqual(C.readComparePrefs(store(JSON.stringify({ mode: 'overlay', opacity: 35 }))), { mode: 'overlay', opacity: 35 });
  assert.deepEqual(C.readComparePrefs(store(JSON.stringify({ mode: 'x', opacity: 'y' }))), { mode: null, opacity: null });
  assert.deepEqual(C.readComparePrefs(store('{not json')), { mode: null, opacity: null });
  assert.deepEqual(C.readComparePrefs(store(null)), { mode: null, opacity: null });
  assert.deepEqual(C.readComparePrefs({ getItem() { throw new Error('blocked'); } }), { mode: null, opacity: null });
  assert.deepEqual(C.readComparePrefs(null), { mode: null, opacity: null });
  const saved = {};
  C.writeComparePrefs({ setItem: (k, v) => (saved[k] = v) }, { mode: 'overlay', opacity: 65 });
  assert.deepEqual(JSON.parse(saved[C.PREFS_KEY]), { mode: 'overlay', opacity: 65 });
  C.writeComparePrefs({ setItem() { throw new Error('quota'); } }, { mode: 'pair', opacity: 50 }); // no throw
  // the URL first, then the stored choice, then pair / 50
  assert.deepEqual(C.comparePrefs({ mode: null, opacity: null }, { cmp: null, opacity: null }), { mode: 'pair', opacity: 50 });
  assert.deepEqual(C.comparePrefs({ mode: 'overlay', opacity: 20 }, { cmp: null, opacity: null }), { mode: 'overlay', opacity: 20 });
  assert.deepEqual(C.comparePrefs({ mode: 'overlay', opacity: 20 }, o('?cmp=pair&opacity=80')), { mode: 'pair', opacity: 80 });
  // the URL while comparing: view and cmp (opacity in overlay mode); the rest kept as written; removed on leaving
  const search = '?model=/files/m.glb&poses=%2Fapi%2Flandmarks%2Flm%2Fposes&theme=dusk';
  assert.equal(C.compareSearch(search, { id: SFM.id, mode: 'pair', opacity: 50 }), `${search}&view=${SFM.id}&cmp=pair`);
  assert.equal(C.compareSearch(search, { id: SFM.id, mode: 'overlay', opacity: 35 }), `${search}&view=${SFM.id}&cmp=overlay&opacity=35`);
  assert.equal(C.compareSearch(search, { id: EXIF.id, mode: 'photo', opacity: 35 }), `${search}&view=${EXIF.id}`);
  assert.equal(C.compareSearch(`${search}&view=${EXIF.id}&cmp=overlay&opacity=10`, null), search);
  assert.equal(C.compareSearch(`?view=${SFM.id}&cmp=pair`, null), '');
  assert.equal(C.compareSearch(`?cmp=pair&view=${SFM.id}&photo=${SFM.id}`, { id: SNAPPED.id, mode: 'pair' }), `?photo=${SFM.id}&view=${SNAPPED.id}&cmp=pair`);
  assert.equal(C.compareSearch(search, { id: STORE_ID, mode: 'pair' }), search, 'never anything but a public id');
  // what compareSearch writes, parsePoseParams reads back
  const back = P.parsePoseParams(C.compareSearch(search, { id: SFM.id, mode: 'overlay', opacity: 35 }), 'https://c.example');
  assert.deepEqual([back.view, back.cmp, back.opacity, back.url], [SFM.id, 'overlay', 35, '/api/landmarks/lm/poses']);
});

// ---------- the DOM (web/tests/fake-dom.mjs) ----------

function setup(options = {}) {
  const doc = installDom();
  const calls = [];
  const view = C.createCompareView(doc, {
    onStep: (d) => calls.push(['step', d]),
    onExit: () => calls.push(['exit']),
    onMode: (m) => calls.push(['mode', m]),
    onOpacity: (v) => calls.push(['opacity', v]),
    onPhoto: (s, size) => calls.push(['photo', s, size]),
    ...options,
  });
  const q = (sel) => view.el.querySelector(sel);
  return { doc, view, calls, q, restore: () => doc.restore() };
}
const row = (ph, k = 3, n = 9) => ({ banner: C.compareBanner(ph, ROUNDING), url: C.photoUrl(POSES, ph.id), alt: `photo ${ph.src}`, k, n });

test('the view: a region with the top bar, the photo and its state, the model chip, the bar; Escape and the arrows', () => {
  const { doc, view, calls, q, restore } = setup({ mode: 'pair', opacity: 50 });
  try {
    assert.equal(view.el.parentNode, doc.body);
    assert.equal(view.el.className, 'pp-cmp');
    assert.equal(view.el.getAttribute('role'), 'region');
    assert.equal(view.el.getAttribute('aria-label'), '照片和素模对照');
    assert.equal(view.el.dataset.mode, 'pair');
    assert.equal(q('.pp-cmp-top').getAttribute('aria-live'), 'polite');
    // its rules: one constructed sheet, however many views are opened; never a <style> element, which the pages'
    // Content-Security-Policy would refuse (D34)
    assert.equal(doc.adoptedStyleSheets.length, 1);
    assert.match(doc.adoptedStyleSheets[0].text, /\.pp-cmp\{/);
    C.createCompareView(doc, {}).destroy();
    assert.equal(doc.adoptedStyleSheets.length, 1);
    assert.deepEqual(doc.head.children.map((n) => n.tagName), []);
    view.show(row(SFM));
    const img = q('.pp-cmp-photo');
    assert.equal(img, view.img);
    assert.equal(img.src, `${POSES}/${SFM.id}/photo`);
    assert.equal(img.alt, 'photo sfm');
    assert.equal(img.decoding, 'async');
    assert.equal(view.el.dataset.photo, 'loading');
    assert.equal(text(q('.pp-cmp-state')), '照片加载中…');
    assert.equal(text(q('.pp-cmp-title')), '照片和从同一位置看到的素模');
    assert.match(text(q('.pp-cmp-detail')), /^来源：模型重建；位置 ±16 米/);
    assert.equal(q('.pp-cmp-warn').hidden, false);
    assert.equal(text(q('.pp-cmp-count')), '第 3 / 9 张');
    const range = q('input');
    assert.equal(range.getAttribute('type'), 'range');
    assert.deepEqual(['min', 'max', 'step'].map((a) => range.getAttribute(a)), ['0', '100', '5']);
    assert.equal(range.getAttribute('aria-label'), '叠加时照片的浓淡');
    assert.equal(range.disabled, true, 'no slider before the photo is there');
    assert.equal(q('.pp-cmp-op').hidden, true, 'the slider only in overlay mode');
    // the photo arrives
    img.naturalWidth = 1600;
    img.naturalHeight = 1203;
    img.dispatch('load');
    assert.equal(view.el.dataset.photo, 'loaded');
    assert.equal(q('.pp-cmp-state').hidden, true);
    assert.equal(range.disabled, false);
    assert.deepEqual(calls.pop(), ['photo', 'loaded', { w: 1600, h: 1203 }]);
    // the chips, the buttons
    assert.deepEqual(view.el.querySelectorAll('.pp-cmp-chip').map(text), ['照片', '素模']);
    assert.equal(text(q('.pp-cmp-back')), '返回总览');
    assert.equal(q('.pp-cmp-prev').getAttribute('aria-label'), '上一张（绕地标逆时针）');
    assert.equal(q('.pp-cmp-next').getAttribute('title'), '下一张（绕地标顺时针）');
    q('.pp-cmp-prev').click();
    q('.pp-cmp-next').click();
    q('.pp-cmp-back').click();
    assert.deepEqual(calls.splice(0), [['step', -1], ['step', 1], ['exit']]);
    // the keyboard: Escape leaves, the arrows step, but not while a form field has the focus
    doc.dispatch('keydown', { key: 'ArrowRight' });
    doc.dispatch('keydown', { key: 'ArrowLeft' });
    for (const tag of ['input', 'select', 'textarea']) doc.dispatch('keydown', { key: 'ArrowRight', target: doc.createElement(tag) });
    doc.dispatch('keydown', { key: 'ArrowRight', target: range });
    doc.dispatch('keydown', { key: 'ArrowRight', ctrlKey: true });
    const esc = doc.dispatch('keydown', { key: 'Escape', target: range });
    assert.equal(esc.defaultPrevented, true);
    doc.dispatch('keydown', { key: 'Enter' });
    assert.deepEqual(calls.splice(0), [['step', 1], ['step', -1], ['exit']]);
    // prev / next off with one photo on the ring
    view.show(row(SFM, 1, 1));
    assert.equal(q('.pp-cmp-prev').disabled, true);
    assert.equal(q('.pp-cmp-next').disabled, true);
    view.show({ ...row(SNAPPED), banner: C.compareBanner(SNAPPED, ROUNDING) });
    assert.equal(q('.pp-cmp-warn').hidden, true);
    // destroy: gone, and so is the key handler
    view.destroy();
    assert.equal(view.el.parentNode, null);
    doc.dispatch('keydown', { key: 'Escape' });
    assert.deepEqual(calls.filter((c) => c[0] !== 'photo'), []);
  } finally {
    restore();
  }
});

test('the view: a failed photo says so and disables the slider; a new row gets a new image; stale events are ignored', () => {
  const { view, calls, q, restore } = setup({ mode: 'overlay', opacity: 50 });
  try {
    view.show(row(SFM));
    const first = view.img;
    first.dispatch('error');
    assert.equal(view.el.dataset.photo, 'missing');
    assert.equal(text(q('.pp-cmp-state')), '这张照片现在显示不了（可能已被撤回）。素模视角照常显示。');
    assert.equal(q('.pp-cmp-state').hidden, false);
    assert.equal(q('input').disabled, true);
    assert.deepEqual(calls.pop(), ['photo', 'missing', null]);
    // the next row: a fresh image, loading; the old one's late events change nothing
    view.show(row(SNAPPED));
    assert.notEqual(view.img, first);
    assert.equal(first.parentNode, null);
    assert.equal(view.img.src, `${POSES}/${SNAPPED.id}/photo`);
    assert.equal(view.el.dataset.photo, 'loading');
    first.dispatch('load');
    first.dispatch('error');
    assert.equal(view.el.dataset.photo, 'loading');
    assert.equal(view.el.querySelectorAll('.pp-cmp-photo').length, 1);
    // no URL (not a public id): missing at once, nothing fetched
    view.show({ ...row(SFM), url: null });
    assert.equal(view.img.src, undefined);
    assert.equal(view.el.dataset.photo, 'missing');
  } finally {
    restore();
  }
});

test('the view: the mode buttons switch data-mode and aria-pressed; the slider sets the photo\'s opacity', () => {
  const { view, calls, q, restore } = setup({ mode: 'pair', opacity: 50 });
  try {
    view.show(row(SFM));
    view.img.dispatch('load');
    calls.splice(0);
    const [pairBtn, overlayBtn] = view.el.querySelectorAll('.pp-cmp-mode');
    assert.deepEqual([text(pairBtn), text(overlayBtn)], ['并排', '叠加']);
    assert.deepEqual([pairBtn.getAttribute('aria-pressed'), overlayBtn.getAttribute('aria-pressed')], ['true', 'false']);
    assert.equal(view.img.style.opacity, '1', 'side by side the photo is opaque');
    overlayBtn.click();
    assert.equal(view.el.dataset.mode, 'overlay');
    assert.deepEqual([pairBtn.getAttribute('aria-pressed'), overlayBtn.getAttribute('aria-pressed')], ['false', 'true']);
    assert.equal(q('.pp-cmp-op').hidden, false);
    assert.equal(view.img.style.opacity, '0.5');
    const range = q('input');
    assert.equal(range.value, '50');
    assert.equal(range.getAttribute('aria-valuetext'), '50%');
    range.value = '30';
    range.dispatch('input');
    assert.equal(view.img.style.opacity, '0.3');
    assert.equal(range.getAttribute('aria-valuetext'), '30%');
    overlayBtn.click(); // already on: nothing
    pairBtn.click();
    assert.equal(view.el.dataset.mode, 'pair');
    assert.equal(view.img.style.opacity, '1');
    assert.equal(q('.pp-cmp-op').hidden, true);
    assert.deepEqual(calls.splice(0), [['mode', 'overlay'], ['opacity', 30], ['mode', 'pair']]);
    // the opacity survives a new row
    view.setMode('overlay');
    view.show(row(SNAPPED));
    assert.equal(view.img.style.opacity, '0.3');
    assert.deepEqual(calls.filter((c) => c[0] !== 'photo'), [], 'setMode from the page reports nothing back');
    // place(): the panes and the photo where the layout says; the pair button names the direction
    const lay = C.compareLayout({ vw: 390, vh: 844, aspect: 1.33, mode: 'pair', ...BARS });
    view.place(lay, { x: lay.photo.x + 0.25, y: lay.photo.y, w: lay.photo.w - 0.5, h: lay.photo.h });
    assert.deepEqual(q('.pp-cmp-pane').style, { left: `${lay.photo.x}px`, top: `${lay.photo.y}px`, width: `${lay.photo.w}px`, height: `${lay.photo.h}px` });
    assert.equal(q('.pp-cmp-model').style.top, `${lay.model.y}px`);
    assert.equal(view.img.style.left, `${lay.photo.x + 0.25}px`);
    assert.equal(text(pairBtn), '上下');
    view.place(C.compareLayout({ vw: 1280, vh: 860, aspect: 1.33, mode: 'pair', ...BARS }), lay.photo);
    assert.equal(text(pairBtn), '并排');
    // the next photo is placed where the last one was until the page lays it out again
    view.show(row(SFM));
    assert.equal(view.img.style.left, `${lay.photo.x}px`);
  } finally {
    restore();
  }
});

test('photo-only: one pane, the honest label and Close; no steps, no modes, no slider; Escape closes', () => {
  const { doc, view, calls, q, restore } = setup({ mode: 'photo' });
  try {
    assert.equal(view.el.dataset.mode, 'photo');
    view.show({ banner: C.compareBanner(EXIF, ROUNDING), url: C.photoUrl(POSES, EXIF.id), alt: 'x' });
    assert.equal(view.img.src, `${POSES}/${EXIF.id}/photo`);
    assert.equal(text(q('.pp-cmp-title')), '只有照片（不能和素模对照）');
    assert.match(text(q('.pp-cmp-detail')), /不能和素模对照。$/);
    assert.equal(q('.pp-cmp-warn').hidden, true);
    assert.equal(text(q('.pp-cmp-back')), '关闭');
    assert.equal(q('.pp-cmp-nav').hidden, true);
    assert.equal(q('.pp-cmp-modes').hidden, true);
    assert.equal(q('.pp-cmp-op').hidden, true);
    view.setMode('overlay'); // ignored
    assert.equal(view.el.dataset.mode, 'photo');
    doc.dispatch('keydown', { key: 'ArrowRight' });
    doc.dispatch('keydown', { key: 'Escape' });
    q('.pp-cmp-back').click();
    assert.deepEqual(calls.filter((c) => c[0] !== 'photo'), [['exit'], ['exit']]);
  } finally {
    restore();
  }
});

test('the display settings (#prefs) sit in the bar while a photo is shown, compared or alone, then go back (D32)', () => {
  for (const mode of ['pair', 'overlay', 'photo']) {
    const doc = installDom();
    try {
      // the page's panel: a title, the settings, the attribution; the panel is hidden while a photo is shown
      const panel = doc.getElementById('panel');
      const title = doc.createElement('h1');
      const prefs = doc.createElement('div');
      prefs.setAttribute('id', 'prefs');
      const lang = doc.createElement('select');
      lang.setAttribute('id', 'pref-lang');
      const theme = doc.createElement('select');
      theme.setAttribute('id', 'pref-theme');
      prefs.append(lang, theme);
      const attribution = doc.createElement('div');
      panel.append(title, prefs, attribution);
      const view = C.createCompareView(doc, { mode, prefs });
      view.show(row(mode === 'photo' ? EXIF : SFM));
      const bar = view.el.querySelector('.pp-cmp-bar');
      assert.equal(prefs.parentNode, bar, `${mode}: the settings are in the bar`);
      assert.equal(bar.children.at(-1), prefs, `${mode}: after the bar's own controls`);
      assert.equal(bar.querySelector('#pref-lang'), lang);
      assert.equal(bar.querySelector('#pref-theme'), theme);
      assert.equal(prefs.hidden, false);
      assert.deepEqual(panel.children, [title, attribution]);
      view.destroy();
      assert.deepEqual(panel.children, [title, prefs, attribution], `${mode}: back in their place`);
      assert.equal(view.el.parentNode, null);
    } finally {
      doc.restore();
    }
  }
  // the bar's CSS keeps them in line (no panel margin) and touch-sized
  const { doc, view, restore } = setup();
  try {
    const css = doc.adoptedStyleSheets[0].text;
    assert.match(css, /\.pp-cmp-bar #prefs\{margin:0\}/);
    assert.match(css, /@media \(pointer:coarse\)\{[^\n]*\.pp-cmp-bar select\{min-height:44px\}/);
    // without settings to borrow, nothing is added to the bar
    view.show(row(SFM));
    assert.equal(view.el.querySelector('#prefs'), null);
    view.destroy();
  } finally {
    restore();
  }
});

test('in English: every text of the view, no Chinese (D32)', () => {
  inLang('en', () => {
    const { view, q, restore } = setup({ mode: 'overlay', opacity: 35 });
    try {
      view.show(row(SFM, 12, 35));
      assert.equal(view.el.getAttribute('aria-label'), 'Photo and model comparison');
      assert.equal(text(q('.pp-cmp-count')), '12 of 35');
      assert.equal(text(q('.pp-cmp-state')), 'Loading photo…');
      assert.deepEqual(view.el.querySelectorAll('.pp-cmp-mode').map(text), ['Side by side', 'Overlay']);
      assert.equal(text(q('.pp-cmp-op')), 'ModelPhoto');
      assert.equal(q('input').getAttribute('aria-label'), 'Photo opacity');
      assert.equal(q('input').getAttribute('aria-valuetext'), '35%');
      assert.equal(text(q('.pp-cmp-back')), 'Back to overview');
      assert.deepEqual(view.el.querySelectorAll('.pp-cmp-chip').map(text), ['Photo', 'Clay model']);
      assert.equal(q('.pp-cmp-prev').getAttribute('aria-label'), 'Previous photo (anticlockwise)');
      assert.equal(q('.pp-cmp-next').getAttribute('aria-label'), 'Next photo (clockwise)');
      view.place(C.compareLayout({ vw: 390, vh: 844, aspect: 1.33, mode: 'overlay', ...BARS }), { x: 0, y: 0, w: 1, h: 1 });
      assert.equal(text(view.el.querySelectorAll('.pp-cmp-mode')[0]), 'Stacked');
      view.img.dispatch('error');
      assert.equal(text(q('.pp-cmp-state')), "This photo can't be shown right now (it may have been withdrawn). The model view is still shown.");
      const shown = [text(view.el), view.el.getAttribute('aria-label'), ...view.el.querySelectorAll('button, input').flatMap((b) => [b.getAttribute('title') ?? '', b.getAttribute('aria-label') ?? ''])];
      for (const s of shown) assert.ok(!CJK.test(s), s);
    } finally {
      restore();
    }
    const photoOnly = setup({ mode: 'photo' });
    try {
      photoOnly.view.show({ banner: C.compareBanner(EXIF, ROUNDING), url: C.photoUrl(POSES, EXIF.id), alt: 'x' });
      assert.equal(text(photoOnly.q('.pp-cmp-back')), 'Close');
      assert.ok(!CJK.test(text(photoOnly.view.el)));
    } finally {
      photoOnly.restore();
    }
  });
  assert.deepEqual([...missing], [], 'every text these tests asked for exists');
});
