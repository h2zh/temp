// Run from the repo root: node --test 'web/tests/*.test.mjs'
// The pure helpers of the clay viewer's photo-pose layer (web/photoposes.js, D30); three.js is only used in a browser.
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { missing } from '../i18n.js';
import * as P from '../photoposes.js';
import { SOURCE_TEXT, parsePoses } from '../poses.js';
import { inLang } from './fake-dom.mjs';

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

// rows as crowd3d/photoposes.py public_row writes them (whole metres and degrees, hfov in 5-degree steps)
const SFM = { id: '0a0a0a0a0a0a', src: 'sfm', p: [84, -4, 2], yaw: 265, pitch: 8, roll: 0, hfov: 40, aspect: 1.76, sp: 16, sy: 21 };
const SNAPPED = { id: '1b1b1b1b1b1b', src: 'sfm', p: [23, -1, 2], yaw: 277, pitch: 11, roll: 0, hfov: 65, aspect: 1.33, sp: 7, sy: 7 };
const EXIF = { id: '2c2c2c2c2c2c', src: 'exif', p: [107, -39, null], yaw: 269, pitch: null, roll: null, hfov: 60, aspect: 1.33, sp: 41, sy: 31 };
const LOC = { id: '3d3d3d3d3d3d', src: 'loc', p: [-30, 12, 1], yaw: 95, pitch: -3, roll: 2, hfov: 55, aspect: 0.75, sp: 17, sy: 22 };
const UNCERTAINTY = {
  sigmas: 2,
  measured_coverage: {
    sfm: { within: 146, n: 165, runs: 2, landmarks: 4, lowest: { within: 1, n: 10 }, highest: { within: 13, n: 13 }, rule_calibrated_here: false },
    loc: null,
    exif: { within: 23, n: 28, runs: 2, landmarks: 3, lowest: { within: 1, n: 2 }, highest: { within: 1, n: 1 }, rule_calibrated_here: true },
  },
};
const ROUNDING = { position_m: 1, angle_deg: 1, fov_deg: 5 };
const doc = (photos, over = {}) => ({
  format: 'crowd3d.photo-poses', version: 2, landmark: { id: 'lm', name: 'X' }, as_of: 1762236000, rounding: ROUNDING,
  uncertainty: UNCERTAINTY, counts: { photos: 9, shared: 6, shown: 4, sfm: 2, loc: 1, exif: 1, not_determined: 2 }, photos, ...over,
});

test('params: only a same-origin poses URL; photo and view take a 12-hex public id only', () => {
  const o = P.parsePoseParams('?poses=/api/landmarks/lm/poses&photo=0a0a0a0a0a0a&view=abc&err=1&cluster=0', 'https://c.example');
  assert.equal(o.url, '/api/landmarks/lm/poses');
  assert.equal(o.select, '0a0a0a0a0a0a');
  assert.equal(o.view, null);
  assert.equal(o.allErrors, true);
  assert.equal(o.cluster, false);
  assert.equal(o.show, true);
  assert.equal(o.fit, 'photos');
  assert.equal(P.parsePoseParams('?poses=https://evil.example/poses', 'https://c.example').url, null);
  assert.equal(P.parsePoseParams('?poses=%2Fapi%2Flandmarks%2Flm%2Fposes', 'https://c.example').url, '/api/landmarks/lm/poses');
  assert.equal(P.parsePoseParams('', 'https://c.example').url, null);
  assert.equal(P.parsePoseParams('?photos=0&fit=model').show, false);
  assert.equal(P.parsePoseParams('?photos=0&fit=model').fit, 'model');
  assert.equal(P.parsePoseParams('?photo=0A0A0A0A0A0A').select, null); // public ids are lower-case hex
});

test('layer input: version 2 only, the same rows as the landmark page, values exactly as the server sent them', () => {
  assert.equal(P.layerInput(null), null);
  assert.equal(P.layerInput(doc([SFM], { version: 1 })), null);
  const bad = [
    { ...SFM, id: '4e4e4e4e4e4e', src: 'gps', yaw: null }, // position only: never drawn
    { ...SFM, id: '5f5f5f5f5f5f', src: 'approx' },
    { ...SFM, id: '6a6a6a6a6a6a', yaw: null },
    { ...SFM, id: '7b7b7b7b7b7b', p: [84.4, -4, 2] }, // not whole metres: not a public row
    { ...SFM, id: '8c8c8c8c8c8c', sp: null }, // never drawn without its range
    { ...SFM, id: '9d9d9d9d9d9d', sy: 21.5 },
  ];
  const d = doc([SFM, SNAPPED, EXIF, LOC, ...bad]);
  const input = P.layerInput(d);
  assert.deepEqual(input.photos.map((r) => r.id), parsePoses(d).photos.map((r) => r.id));
  assert.deepEqual(input.photos.map((r) => r.id), [SFM.id, SNAPPED.id, EXIF.id, LOC.id]);
  assert.equal(input.dropped, bad.length);
  // no client-side rounding or defaults: every value is the server's
  for (const [row, src] of [[input.photos[0], SFM], [input.photos[2], EXIF], [input.photos[3], LOC]]) {
    assert.deepEqual(row, src);
  }
  assert.equal(input.asOf, 1762236000);
  assert.deepEqual(input.counts, { photos: 9, shared: 6, shown: 4, not_determined: 2, sfm: 2, loc: 1, exif: 1 });
  assert.deepEqual(input.rounding, ROUNDING);
  // only the public keys reach the layer, whatever else a document carries
  const extra = P.layerInput(doc([{ ...SFM, t: '2019-07', user_id: 'alice', thumb: { url: '/x.jpg' }, own: true, approx: true }]));
  assert.deepEqual(Object.keys(extra.photos[0]).sort(), ['aspect', 'hfov', 'id', 'p', 'pitch', 'roll', 'sp', 'src', 'sy', 'yaw']);
  // an implausible optional value is unknown (null), never a default
  const odd = P.layerInput(doc([{ ...SFM, pitch: 120, roll: 'x', hfov: 500, aspect: 0 }])).photos[0];
  assert.deepEqual([odd.pitch, odd.roll, odd.hfov, odd.aspect], [null, null, null, null]);
  const odd2 = P.layerInput(doc([{ ...SFM, hfov: 62.5 }])).photos[0];
  assert.equal(odd2.hfov, null); // the server sends whole degrees in 5-degree steps
  // malformed counts are unknown, not zero
  assert.equal(P.layerInput(doc([SFM], { counts: { shared: -1, photos: '9' } })).counts.shared, null);
});

test('glyphs: a frustum only with a measured height, pitch, roll, field of view and aspect; EXIF on the ground', () => {
  assert.equal(P.glyphOf(SFM), 'frustum');
  assert.equal(P.glyphOf(LOC), 'frustum');
  assert.equal(P.glyphOf(SNAPPED), 'frustum');
  assert.equal(P.glyphOf(EXIF), 'wedge');
  assert.equal(P.glyphOf({ ...EXIF, p: [1, 2, 3], pitch: 0, roll: 0 }), 'wedge'); // an EXIF row is never a frustum
  for (const k of ['pitch', 'roll', 'hfov', 'aspect']) assert.equal(P.glyphOf({ ...SFM, [k]: null }), 'wedge', k);
  assert.equal(P.glyphOf({ ...SFM, p: [84, -4, null] }), 'wedge');
  // an EXIF row's missing height is drawn on the ground, not at an assumed camera height
  assert.deepEqual(P.enuToThree(EXIF.p), [107, 0, 39]);
  assert.deepEqual(P.enuToThree(SFM.p), [84, 2, 4]);
});

test('the 2-sigma disc is drawn whenever sp > 10 m, else when the row is picked or every range is on', () => {
  assert.equal(P.DISC_SP_M, 10);
  assert.equal(P.showsDisc(SFM), true);
  assert.equal(P.showsDisc(EXIF), true);
  assert.equal(P.showsDisc(LOC), true);
  assert.equal(P.showsDisc(SNAPPED), false);
  assert.equal(P.showsDisc({ ...SNAPPED, sp: 10 }), false);
  assert.equal(P.showsDisc({ ...SNAPPED, sp: 11 }), true);
  assert.equal(P.showsDisc(SNAPPED, { selected: true }), true);
  assert.equal(P.showsDisc(SNAPPED, { all: true }), true);
  const d = P.discGeometry(10, -20, 16, 0.04, 48);
  assert.equal(d.tri.length, 48 * 9);
  for (let k = 0; k < d.tri.length; k += 9) {
    near(Math.hypot(d.tri[k + 3] - 10, d.tri[k + 5] + 20), 16, 1e-9);
    near(d.tri[k + 4], 0.04);
  }
  // the legend says so only when such a row is drawn
  assert.match(P.legendNotes(P.layerInput(doc([SFM]))).join('\n'), /范围超过 10 米的照片在地上画出范围圆（2σ）/);
  assert.doesNotMatch(P.legendNotes(P.layerInput(doc([SNAPPED]))).join('\n'), /范围圆/);
});

test('"view from this photo" needs a real pose and field of view: never for EXIF, never with an assumed value', () => {
  assert.equal(P.viewFromBlocked(SFM), null);
  assert.equal(P.viewFromBlocked(LOC), null);
  // a code; BLOCKED_TEXT says it in words (D32)
  const blocked = (row) => P.BLOCKED_TEXT[P.viewFromBlocked(row)];
  assert.equal(P.viewFromBlocked(EXIF), 'exif');
  assert.match(blocked(EXIF), /手机 GPS \+ 罗盘.*没有测出拍摄高度和仰角/);
  assert.equal(P.viewFromBlocked({ ...EXIF, p: [1, 2, 3], pitch: 0, roll: 0 }), 'exif');
  assert.match(blocked({ ...EXIF, p: [1, 2, 3], pitch: 0, roll: 0 }), /手机 GPS/);
  assert.equal(P.viewFromBlocked({ ...SFM, hfov: null }), 'fov');
  assert.match(blocked({ ...SFM, hfov: null }), /没有测出视角/);
  assert.match(blocked({ ...SFM, aspect: null }), /没有测出视角/);
  assert.equal(P.viewFromBlocked({ ...SFM, pitch: null }), 'pose');
  assert.match(blocked({ ...SFM, pitch: null }), /没有测出拍摄高度、仰角或侧倾/);
  assert.match(blocked({ ...SFM, p: [84, -4, null] }), /没有测出拍摄高度/);
  assert.deepEqual(P.BLOCKED, ['exif', 'pose', 'fov']);
  assert.equal(P.viewPose(EXIF, 1280, 860), null);
  assert.equal(P.viewPose({ ...SFM, hfov: null }, 1280, 860), null);
  // the camera goes to the row's own position, direction and field of view
  const v = P.viewPose(SFM, 1280, 860);
  assert.deepEqual(v.position, [84, 2, 4]);
  const b = P.viewBasis(265, 8, 0);
  v.forward.forEach((x, i) => near(x, b.forward[i]));
  near(v.fov, P.photoFrame(40, 1.76, 1280, 860).fov);
  near(v.frame.w, 1);
  near(v.frame.h, 1280 / 860 / 1.76);
  // a looking direction 8 deg up and to the west (-x)
  assert.ok(v.forward[0] < -0.98 && v.forward[1] > 0.13);
});

test('the card: source, position and heading with their ranges, the rounding and the source\'s measured coverage', () => {
  const text = (row) => P.describePhoto(row, { uncertainty: UNCERTAINTY, rounding: ROUNDING }).map((l) => l.v).join('\n');
  const s = text(SFM);
  assert.match(s, /来源：模型重建/);
  assert.match(s, /位置：离地标中心东 84 米，南 4 米；±16 米/);
  assert.match(s, /朝向：265°（西），±21°，仰角 8°/);
  assert.match(s, /视角：水平约 40°/);
  assert.match(s, /位置取整到 1 米、朝向取整到 1°、视角取整到 5°/);
  assert.match(s, /^一张公开的照片$/m);
  assert.match(s, /±范围是 2σ。.*4 个测试地标、2 次运行合计 165 个中 146 个（88%），单个地标单次运行最低 10 个中 1 个（10%）/);
  assert.match(s, /范围规则没有用这些地标调过/);
  assert.doesNotMatch(s, /精确|不是独立检验/);
  const e = text(EXIF);
  assert.match(e, /来源：手机 GPS \+ 罗盘（未经模型核对）/);
  assert.match(e, /（拍摄高度没有测出）；±41 米/);
  assert.match(e, /±31°，仰角没有测出/);
  assert.match(e, /28 个中 23 个（82%）.*不是独立检验/);
  assert.match(text(LOC), /与模型比对.*还没有实测/s);
  assert.match(text(LOC), /俯角 3°/);
  // nothing about who took it or when, whatever the row carries
  const extra = text({ ...SFM, t: '2019-07', user_id: 'alice' });
  assert.doesNotMatch(extra, /alice|2019|拍摄者：/);
  assert.match(extra, /不显示是谁拍的、什么时候拍的；照片本身去掉拍摄信息（EXIF）并缩小后公开。$/);
  // without the server's measurement nothing is claimed
  assert.match(P.describePhoto(SFM).map((l) => l.v).join('\n'), /模型重建[^]*还没有实测/);
});

test('banner, tip text and legend ranges carry the 2-sigma ranges; the legend counts are the server\'s', () => {
  const b = P.viewBanner(SFM, ROUNDING);
  assert.match(b, /^从这张照片的位置和视角看（模型重建）：位置 ±16 米，朝向 ±21°（2σ），位置取整到 1 米、朝向取整到 1°、视角取整到 5°/);
  assert.match(b, /看到的画面可能和照片对不上$/);
  assert.doesNotMatch(P.viewBanner(SNAPPED, ROUNDING), /对不上/);
  assert.equal(P.errorText(EXIF), '位置 ±41 米，朝向 ±31°');
  assert.equal(P.errorRange([SFM, SNAPPED, EXIF], 'sfm'), '±7–16 米 / ±7–21°');
  assert.equal(P.errorRange([SFM, SNAPPED, EXIF], 'exif'), '±41 米 / ±31°');
  assert.equal(P.errorRange([SFM], 'loc'), '');
  const input = P.layerInput(doc([SFM, EXIF]));
  const t = P.legendTotals(input);
  assert.deepEqual(t.bySource, { sfm: 2, loc: 1, exif: 1 }); // the server's counts over the snapshot
  assert.deepEqual(t.drawn, { sfm: 1, loc: 0, exif: 1 });
  assert.equal(t.shown, 4);
  const bare = P.legendTotals(P.layerInput(doc([SFM, EXIF], { counts: undefined })));
  assert.deepEqual(bare.bySource, { sfm: 1, loc: 0, exif: 1 });
  assert.equal(bare.shown, 2);
  const notes = P.legendNotes({ ...input, dropped: 2 }).join('\n');
  assert.match(notes, /模型重建：4 个测试地标/);
  assert.match(notes, /手机 GPS \+ 罗盘（未经模型核对）：3 个测试地标.*不是独立检验/);
  assert.match(notes, /位置取整到 1 米、朝向取整到 1°、视角取整到 5°；不显示是谁拍的、什么时候拍的；照片本身去掉拍摄信息（EXIF）并缩小后公开。/);
  assert.doesNotMatch(notes, /同意/);
  assert.match(notes, /\d{4}-\d{2}-\d{2} \d{2}:\d{2} 更新（这里每次建模结束后更新）/);
  assert.match(notes, /2 行数据不完整，没有画/);
  assert.equal(P.asOfText(null), '');
  // the field-of-view step comes from the server (rounding.fov_deg), 5 deg without it (D30 rounding)
  assert.equal(P.roundingText(null), '位置取整到 1 米、朝向取整到 1°、视角取整到 5°');
  assert.equal(P.roundingText({ position_m: 2, angle_deg: 3, fov_deg: 10 }), '位置取整到 2 米、朝向取整到 3°、视角取整到 10°');
  assert.equal(SOURCE_TEXT.exif, '手机 GPS + 罗盘（未经模型核对）');
});

test('basis: north-looking level camera; pitch and roll; ENU -> three axes', () => {
  const b = P.viewBasis(0, 0, 0);
  [0, 0, -1].forEach((v, i) => near(b.forward[i], v)); // north = -z
  [1, 0, 0].forEach((v, i) => near(b.right[i], v));
  [0, 1, 0].forEach((v, i) => near(b.up[i], v));
  const e = P.viewBasis(90, 30, 0); // east, 30 deg up
  near(e.forward[0], Math.cos(Math.PI / 6));
  near(e.forward[1], 0.5);
  const r = P.viewBasis(0, 0, 90); // right side raised 90 deg: right points up
  [0, 1, 0].forEach((v, i) => near(r.right[i], v, 1e-9));
  const c = P.viewBasis(123, -17, 8); // orthonormal, right-handed (right x up = -forward)
  const dot = (a, b2) => a.reduce((s, v, i) => s + v * b2[i], 0);
  near(dot(c.forward, c.right), 0, 1e-12);
  near(dot(c.forward, c.up), 0, 1e-12);
  const x = [c.right[1] * c.up[2] - c.right[2] * c.up[1], c.right[2] * c.up[0] - c.right[0] * c.up[2], c.right[0] * c.up[1] - c.right[1] * c.up[0]];
  x.forEach((v, i) => near(v, -c.forward[i], 1e-12));
});

test('fov and photo frame in a viewport', () => {
  near(P.vfovOf(90, 1), 90);
  const wide = P.photoFrame(60, 1.5, 1600, 800); // viewport 2:1 wider than a 3:2 photo -> heights match
  near(wide.fov, P.vfovOf(60, 1.5));
  near(wide.frame.h, 1);
  near(wide.frame.w, 0.75);
  const tall = P.photoFrame(60, 1.5, 400, 800); // phone portrait viewport
  near(tall.frame.w, 1);
  near(tall.frame.h, 0.5 / 1.5);
  assert.ok(tall.fov > wide.fov);
});

test('screen clusters: greedy by priority, radius inclusive; framing keeps the nearest 90 %', () => {
  const items = [{ i: 0, x: 0, y: 0, prio: 2 }, { i: 1, x: 10, y: 0, prio: 0 }, { i: 2, x: 31, y: 0, prio: 1 }, { i: 3, x: 100, y: 0, prio: 0 }];
  const c = P.clusterScreen(items, 20);
  assert.deepEqual(c.map((k) => k.members.sort()), [[0, 1], [3], [2]]);
  assert.equal(c[0].seed, 1);
  const rows = Array.from({ length: 10 }, (_, k) => ({ p: [10 * (k + 1), 0, 2] }));
  assert.equal(P.framingSet(rows).length, 9);
  assert.deepEqual(P.framingSet([]), []);
});

test('ground wedge: yaw is a compass bearing (east = +x, north = -z), spanning yaw +- half', () => {
  const bearing = (x, z) => ((Math.atan2(x, -z) * 180) / Math.PI + 360) % 360;
  const w = P.wedgeGeometry(0, 0, 45, 20, 7, 0.05);
  assert.equal(w.tri.length % 9, 0);
  for (let k = 0; k < w.tri.length; k += 3) near(w.tri[k + 1], 0.05);
  const arc = [];
  for (let k = 0; k < w.tri.length; k += 9) {
    near(w.tri[k], 0);
    near(w.tri[k + 2], 0);
    near(Math.hypot(w.tri[k + 3], w.tri[k + 5]), 7, 1e-9);
    arc.push(bearing(w.tri[k + 3], w.tri[k + 5]), bearing(w.tri[k + 6], w.tri[k + 8]));
  }
  for (const a of arc) assert.ok(a >= 25 - 1e-9 && a <= 65 + 1e-9, `arc bearing ${a}`);
  near(arc[0], 25, 1e-9);
  near(arc.at(-1), 65, 1e-9);
  const e = P.wedgeGeometry(10, 20, 90, 10, 5).line.slice(-12); // the two apex -> edge sides
  assert.ok((e[3] + e[9]) / 2 - 10 > 4.8 && Math.abs((e[5] + e[11]) / 2 - 20) < 1e-9, 'yaw 90 points to +x');
  assert.ok(P.WEDGE_MIN_HALF_DEG <= Math.min(SFM.sy, EXIF.sy));
});

// ---------- English (D32) ----------

// Chinese characters and CJK / full-width punctuation
const CJK = /[\p{Script=Han}　-〿＀-￯]/u;
const noCjk = (lines, what) => {
  for (const l of [].concat(lines)) assert.ok(!CJK.test(l), `${what}: ${l}`);
};

test('in English: the legend says the rounding, what is never shown, each source and its counts; no Chinese (D32)', () => {
  inLang('en', () => {
    const input = P.layerInput(doc([SFM, SNAPPED, EXIF, LOC, { ...SFM, id: 'ffffffffffff', src: 'gps' }]));
    const { label, html } = P.legendMarkup(input, { allErrors: false, shown: true });
    assert.equal(label, 'Legend of the photo positions');
    noCjk(html, 'legend');
    assert.match(html, /^<summary><h2>Photo positions and headings<\/h2><\/summary>/);
    assert.match(html, /<p class="pp-count">Public: 6 of 9 photos, 4 shown; 2 not shown \(position and heading could not be determined reliably\)<\/p>/);
    assert.match(html, /3D reconstruction<span class="pp-err">±7–16 m \/ ±7–21°<\/span><span class="pp-n">2<\/span>/);
    assert.match(html, /Matched to the model<span class="pp-err">±17 m \/ ±22°<\/span><span class="pp-n">1<\/span>/);
    // the EXIF source keeps the owner's full wording: the legend shows the label without its hint
    assert.match(html, /Phone GPS and compass \(not checked against the model\)<span class="pp-err">±41 m \/ ±31°<\/span><span class="pp-n">1<\/span>/);
    assert.match(html, /Photos with a range over 10 m have their range circle \(2σ\) drawn on the ground: the camera may be anywhere in the circle/);
    assert.match(html, /Phone GPS and compass \(not checked against the model\): 23 of 28 \(82%\) on 3 test landmarks over 2 runs; .*not an independent check/);
    assert.match(html, /<p class="pp-note">Shown with the position rounded to 1 m, heading to 1°, field of view to 5°\. Who took the photo and when are not shown; the photo itself is public, with its capture data \(EXIF\) removed and scaled down\.<\/p>/);
    assert.match(html, /<p class="pp-note">Updated [A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}\s[AP]M \(after each build here\)<\/p>/);
    assert.match(html, /<p class="pp-note">1 photo not drawn: its data is incomplete<\/p>/);
    assert.match(html, /<input type="checkbox" data-err >Show every photo&#39;s range<\/label><label><input type="checkbox" data-show checked>Show the photos on the model<\/label>$/);
    assert.doesNotMatch(html, /agree|consent/i);
    // the server's counts beside the rows this page could draw; without the server's counts, the rows shown
    const partial = P.legendMarkup(P.layerInput(doc([SFM, EXIF])), {}).html;
    assert.match(partial, /3D reconstruction<span class="pp-err">±16 m \/ ±21°<\/span><span class="pp-n">2 · 1 drawn<\/span>/);
    assert.match(partial, /data-src="loc" checked disabled>.*Matched to the model<span class="pp-n">1 · 0 drawn<\/span>/);
    assert.match(partial, /\(not checked against the model\)<span class="pp-err">±41 m \/ ±31°<\/span><span class="pp-n">1<\/span>/);
    const bare = P.legendMarkup(P.layerInput(doc([SFM, EXIF], { counts: undefined, as_of: null })), {}).html;
    assert.match(bare, /<p class="pp-count">Showing 2 photos<\/p>/);
    assert.doesNotMatch(bare, /Updated/);
  });
});

test('in English: the card of an sfm and an EXIF row, why "view from this photo" is off, the tip and the banner (D32)', () => {
  inLang('en', () => {
    const card = (row) => P.describePhoto(row, { uncertainty: UNCERTAINTY, rounding: ROUNDING }).map((l) => l.v);
    const s = card(SFM);
    assert.deepEqual(s.slice(0, 5), [
      'A public photo',
      'Source: 3D reconstruction. The photo was used to build the 3D model, which worked out its position and heading.',
      'Position: 84 m east, 4 m south of the landmark centre; ±16 m',
      'Heading: 265° (W), ±21°, tilted 8° up',
      'Field of view: about 40° across',
    ]);
    assert.match(s[5], /^± ranges are 2σ\. How often the true position and heading both fell within the range for this source, measured on the benchmark test landmarks: 146 of 165 \(88%\) on 4 test landmarks over 2 runs; for a single landmark and run, lowest 1 of 10 \(10%\)/);
    assert.match(s[5], /range rule was not tuned on these landmarks\.$/);
    assert.equal(s[6], 'Shown with the position rounded to 1 m, heading to 1°, field of view to 5°. Who took the photo and when are not shown; the photo itself is public, with its capture data (EXIF) removed and scaled down.');
    const e = card(EXIF);
    assert.equal(e[1], 'Source: Phone GPS and compass (not checked against the model). The location and compass heading the phone saved with the photo, not checked against the model; the range is much wider.');
    assert.equal(e[2], 'Position: 107 m east, 39 m south of the landmark centre (the camera height was not measured); ±41 m');
    assert.equal(e[3], 'Heading: 269° (W), ±31°; the tilt was not measured');
    assert.match(e[5], /23 of 28 \(82%\).*not an independent check and the real share may be lower\.$/);
    const l = card(LOC);
    assert.equal(l[3], 'Heading: 95° (E), ±22°, tilted 3° down');
    assert.match(l[5], /: not measured yet\.$/);
    assert.equal(card({ ...SFM, p: [84, -4, 12] })[2], 'Position: 84 m east, 4 m south of the landmark centre, at a height of 12 m; ±16 m');
    assert.equal(card({ ...SFM, hfov: null })[4], 'Field of view: not measured');
    // nothing about who took it or when, whatever the row carries
    const extra = card({ ...SFM, t: '2019-07', user_id: 'alice' }).join('\n');
    assert.doesNotMatch(extra, /alice|2019|taken by|uploaded by/i);
    for (const row of [SFM, SNAPPED, EXIF, LOC, { ...SFM, pitch: null, hfov: null, p: [-3, 0, null] }]) noCjk(card(row), row.id);
    // "view from this photo": the three reasons it is off (D30: never from an assumed height, tilt or field of view)
    assert.deepEqual({ ...P.BLOCKED_TEXT }, {
      exif: "the phone's GPS and compass give only a rough position and heading; the camera's height and tilt weren't measured",
      pose: "the camera's height, tilt or roll wasn't measured",
      fov: "the field of view wasn't measured (unknown focal length or frame size)",
    });
    assert.deepEqual(P.cardActions(SFM), { viewFrom: { label: 'View from this photo', disabled: false, title: null }, blocked: null, close: 'Close' });
    const ex = P.cardActions(EXIF);
    assert.equal(ex.viewFrom.disabled, true);
    assert.equal(ex.viewFrom.title, P.BLOCKED_TEXT.exif);
    assert.equal(ex.blocked, "Can't view from this photo: the phone's GPS and compass give only a rough position and heading; the camera's height and tilt weren't measured.");
    assert.equal(P.cardActions({ ...SFM, roll: null }).blocked, "Can't view from this photo: the camera's height, tilt or roll wasn't measured.");
    assert.equal(P.cardActions({ ...SFM, aspect: null }).blocked, "Can't view from this photo: the field of view wasn't measured (unknown focal length or frame size).");
    // the tip, a cluster's tooltip, the banner, the ranges and the rounding
    assert.equal(P.tipText(EXIF), 'Phone GPS and compass (not checked against the model) · heading 269° (W) · position ±41 m, heading ±31°');
    assert.equal(P.bubbleTitle({ sfm: 3, exif: 1 }), '3D reconstruction: 3 · Phone GPS and compass (not checked against the model): 1 (tap to zoom in)');
    assert.equal(P.viewBanner(SNAPPED, ROUNDING), "From this photo's position and field of view (3D reconstruction): position ±7 m, heading ±7° (2σ); position rounded to 1 m, heading to 1°, field of view to 5°.");
    assert.match(P.viewBanner(SFM, ROUNDING), /\. The camera may be anywhere in this range, so the view may not match the photo\.$/);
    assert.equal(P.errorText(EXIF), 'position ±41 m, heading ±31°');
    assert.equal(P.errorRange([SFM, SNAPPED, EXIF], 'sfm'), '±7–16 m / ±7–21°');
    assert.equal(P.roundingText(null), 'position rounded to 1 m, heading to 1°, field of view to 5°');
    assert.equal(P.roundingText({ position_m: 2, angle_deg: 3, fov_deg: 10 }), 'position rounded to 2 m, heading to 3°, field of view to 10°');
    assert.match(P.asOfText(1762236000), /^[A-Z][a-z]{2} \d{1,2}, 2025, \d{1,2}:\d{2}\s[AP]M$/);
  });
  // Chinese: the same codes, the Chinese reasons
  assert.equal(P.cardActions(EXIF).blocked, `不能从这张照片的角度看：${P.BLOCKED_TEXT.exif}。`);
  assert.equal(P.SOURCE_LABEL.exif, SOURCE_TEXT.exif);
  assert.deepEqual([...missing], [], 'every text these tests asked for exists');
});
