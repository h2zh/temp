// Run from the repo root: node --test 'web/tests/*.test.mjs'
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { compassSvg, escapeXml, polar, sectorName, wedgePath } from '../compass.js';

const LABELS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];

function sectors(spec, labels = LABELS) {
  return spec.map(([count, state, hasFace = false], i) => ({
    index: i,
    label: labels[i],
    count,
    state,
    bearing_center_deg: (i * 360) / spec.length,
    has_face_model: hasFace,
  }));
}

const MIXED = sectors([
  [5, 'face_ready', true],
  [4, 'face_ready'],
  [2, 'partial'],
  [0, 'locked'],
  [1, 'partial'],
  [0, 'locked'],
  [6, 'face_ready', true],
  [0, 'locked'],
]);

function groups(svg) {
  return [...svg.matchAll(/<g class="compass-sector" data-index="(\d+)" data-state="(\w+)">(.*?)<\/g>/g)].map((m) => ({
    index: Number(m[1]),
    state: m[2],
    body: m[3],
  }));
}

function countAnchor(body) {
  const m = body.match(/<text class="compass-count[^"]*" x="([-\d.]+)" y="([-\d.]+)"[^>]*>(\d+)<\/text>/);
  return m ? { x: Number(m[1]), y: Number(m[2]), text: m[3] } : null;
}

function wedgeStart(body) {
  const m = body.match(/<path class="compass-wedge[^"]*" d="M ([-\d.]+) ([-\d.]+) A/);
  return [Number(m[1]), Number(m[2])];
}

test('one wedge per sector, coloured by state class', () => {
  const svg = compassSvg(MIXED);
  const gs = groups(svg);
  assert.equal(gs.length, 8);
  assert.deepEqual(
    gs.map((g) => g.state),
    MIXED.map((s) => s.state),
  );
  for (const g of gs) assert.match(g.body, new RegExp(`class="compass-wedge is-${g.state}"`));
  assert.equal((svg.match(/class="compass-wedge is-locked"/g) || []).length, 3);
  assert.equal((svg.match(/class="compass-wedge is-partial"/g) || []).length, 2);
  assert.equal((svg.match(/class="compass-wedge is-face_ready"/g) || []).length, 3);
});

test('check mark only on wedges with a face model', () => {
  const gs = groups(compassSvg(MIXED));
  const withCheck = gs.filter((g) => g.body.includes('compass-check')).map((g) => g.index);
  assert.deepEqual(withCheck, [0, 6]);
});

test('partial and face-ready wedges show their count, locked ones do not', () => {
  const gs = groups(compassSvg(MIXED));
  for (const g of gs) {
    const anchor = countAnchor(g.body);
    if (g.state === 'locked') assert.equal(anchor, null);
    else assert.equal(anchor.text, String(MIXED[g.index].count));
  }
});

test('north is at the top and indices run clockwise', () => {
  const gs = groups(compassSvg(MIXED));
  const n = countAnchor(gs[0].body);
  const e = countAnchor(gs[2].body);
  const w = countAnchor(gs[6].body);
  assert.ok(Math.abs(n.x - 100) < 1e-6 && n.y < 100, 'sector 0 above the centre');
  assert.ok(e.x > 100 && Math.abs(e.y - 100) < 1e-6, 'sector 2 (E) right of the centre');
  assert.ok(w.x < 100 && Math.abs(w.y - 100) < 1e-6, 'sector 6 (W) left of the centre');
  // Wedge 0 straddles north: it starts left of the top (bearing -22.5) on the outer ring.
  const [x0, y0] = wedgeStart(gs[0].body);
  assert.ok(x0 < 100 && y0 < 100);
});

test('wedges tile the circle with no gaps', () => {
  const gs = groups(compassSvg(MIXED));
  for (let i = 0; i < gs.length; i++) {
    const d = gs[i].body.match(/d="(M [^"]+)"/)[1];
    const arcEnd = d.match(/A 74 74 0 0 1 ([-\d.]+) ([-\d.]+)/);
    const next = wedgeStart(gs[(i + 1) % gs.length].body);
    assert.ok(Math.abs(Number(arcEnd[1]) - next[0]) < 0.02 && Math.abs(Number(arcEnd[2]) - next[1]) < 0.02);
  }
});

test('four sectors give quarter wedges labelled in Chinese', () => {
  const four = sectors(
    [
      [4, 'face_ready', true],
      [1, 'partial'],
      [0, 'locked'],
      [0, 'locked'],
    ],
    ['N', 'E', 'S', 'W'],
  );
  const svg = compassSvg(four);
  assert.equal(groups(svg).length, 4);
  const [x0, y0] = wedgeStart(groups(svg)[0].body);
  const [ex, ey] = polar(-45, 74);
  assert.ok(Math.abs(x0 - ex) < 0.01 && Math.abs(y0 - ey) < 0.01);
  for (const name of ['北', '东', '南', '西']) assert.ok(svg.includes(`>${name}</text>`), name);
  assert.match(svg, /class="compass-label is-north"[^>]*>北</);
});

test('accessible summary and per-wedge tooltips', () => {
  const svg = compassSvg(MIXED, { faceMin: 4 });
  assert.match(svg, /^<svg [^>]*role="img"/);
  assert.match(svg, /aria-label="覆盖罗盘：8 个方向中 3 个照片已够，共 18 张照片"/);
  assert.ok(svg.includes('<title>北（N）：5 张 / 4，照片已够，已生成立面模型</title>'));
  assert.ok(svg.includes('<title>东南（SE）：0 张 / 4，暂无照片</title>'));
  assert.ok(svg.includes('>18</text>'), 'total in the centre');
});

test('compact variant has no text but keeps check marks and a north tick', () => {
  const svg = compassSvg(MIXED, { size: 64, compact: true });
  assert.match(svg, /width="64" height="64"/);
  assert.ok(svg.includes('compass--compact'));
  assert.ok(!svg.includes('<text'));
  assert.equal((svg.match(/compass-check/g) || []).length, 2);
  assert.ok(svg.includes('compass-north'));
});

test('labels and unknown states are escaped / normalised', () => {
  const odd = [
    { index: 0, label: '<b>&"', count: 2, state: 'weird', bearing_center_deg: 0, has_face_model: false },
    { index: 1, label: 'S1', count: 1, state: 'partial', bearing_center_deg: 180, has_face_model: false },
  ];
  const svg = compassSvg(odd);
  assert.ok(!svg.includes('<b>'));
  assert.ok(svg.includes('&lt;b&gt;&amp;&quot;'));
  assert.equal(groups(svg)[0].state, 'locked');
  assert.ok(svg.includes('>S1</text>'));
});

test('empty input renders a placeholder ring', () => {
  for (const input of [[], null, undefined]) {
    const svg = compassSvg(input);
    assert.match(svg, /aria-label="暂无覆盖数据"/);
    assert.match(svg, /<circle class="compass-wedge is-locked"/);
  }
});

test('missing bearing_center_deg falls back to the index', () => {
  const s = MIXED.map(({ bearing_center_deg, ...rest }) => rest);
  assert.equal(compassSvg(s), compassSvg(MIXED));
});

test('helpers', () => {
  assert.deepEqual(polar(0, 10), [100, 90]);
  const [x, y] = polar(90, 10);
  assert.ok(Math.abs(x - 110) < 1e-9 && Math.abs(y - 100) < 1e-9);
  assert.equal(sectorName('NE'), '东北');
  assert.equal(sectorName('S3'), 'S3');
  assert.equal(escapeXml(`a<b>&'"`), 'a&lt;b&gt;&amp;&#39;&quot;');
  assert.match(wedgePath(0, 200, 10, 20), / 0 1 1 /, 'large-arc flag for wedges over 180 degrees');
});

test('failed facades get a "!" badge and say so in their tooltip', () => {
  const svg = compassSvg(MIXED, { faceMin: 4, failed: new Set([1, 6]) });
  const gs = groups(svg);
  // Sector 1 has no model yet: badge instead of a check. Sector 6 keeps its model's check.
  assert.ok(gs[1].body.includes('compass-fail-badge'));
  assert.ok(!gs[1].body.includes('compass-check'));
  assert.ok(gs[1].body.includes('<title>东北（NE）：4 张 / 4，照片已够，立面生成失败</title>'));
  assert.ok(gs[6].body.includes('compass-check') && !gs[6].body.includes('compass-fail'));
  assert.ok(gs[6].body.includes('，已生成立面模型，最近一次更新失败</title>'));
  for (const g of gs.filter((g) => g.index !== 1)) assert.ok(!g.body.includes('compass-fail-badge'), `sector ${g.index}`);
  assert.equal(compassSvg(MIXED, { failed: [] }), compassSvg(MIXED), 'no failures, unchanged markup');
});
