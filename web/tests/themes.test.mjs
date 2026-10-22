// Run from the repo root: node --test 'web/tests/*.test.mjs'
// Viewer themes (web/themes.js, web/themes/*.json): schema, honesty invariant, colour remapping, loading.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  DEFAULT_THEME,
  MIN_CLAY_DELTA_E,
  applyRule,
  buildColorMap,
  deltaE00,
  deltaE00Rgb,
  hexToRgb,
  loadTheme,
  remapPixels,
  resolveTheme,
  rgbKey,
  rgbToHex,
  rgbToLab,
  themeCss,
  validateTheme,
} from '../themes.js';

const DIR = new URL('../themes/', import.meta.url);
const readJson = (name) => JSON.parse(readFileSync(new URL(name, DIR), 'utf8'));
const TABLE = readJson('palette.json').colors;
const THEME_FILES = readdirSync(DIR).filter((f) => f.endsWith('.json') && !['palette.json', 'index.json'].includes(f));
const clone = (o) => JSON.parse(JSON.stringify(o));
const near = (a, b, eps) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('CIEDE2000 matches published reference pairs (Sharma, Wu and Dalal 2005)', () => {
  near(deltaE00([50, 2.6772, -79.7751], [50, 0, -82.7485]), 2.0425, 1e-4);
  near(deltaE00([50, 3.1571, -77.2803], [50, 0, -82.7485]), 2.8615, 1e-4);
  near(deltaE00([50, 2.5, 0], [73, 25, -18]), 27.1492, 1e-4);
  near(deltaE00([2.0776, 0.0795, -1.135], [0.9033, -0.0636, -0.5514]), 0.9082, 1e-4);
  near(rgbToLab([255, 255, 255])[0], 100, 1e-3);
  assert.equal(rgbToHex(hexToRgb('#2d2a32')), '#2d2a32');
});

test('every theme in web/themes is valid, honest and listed in the picker index', () => {
  assert.ok(THEME_FILES.length >= 4, THEME_FILES.join(', '));
  for (const file of THEME_FILES) {
    const theme = readJson(file);
    assert.equal(`${theme.id}.json`, file);
    const v = validateTheme(theme, TABLE);
    assert.ok(v.ok, `${file}: ${v.errors.join('; ')}`);
    assert.ok(v.minClayDeltaE >= MIN_CLAY_DELTA_E, `${file}: ${v.minClayDeltaE}`);
    // every colour of the table gets exactly one colour under the theme
    const { map } = buildColorMap(theme, TABLE);
    assert.equal(map.size, TABLE.length);
  }
  assert.deepEqual([...readJson('index.json').themes].sort(), THEME_FILES.map((f) => f.replace('.json', '')).sort());
  assert.equal(readJson('index.json').themes[0], 'default');
  assert.deepEqual(readJson('default.json'), clone(DEFAULT_THEME));
  for (const id of ['storybook', 'dusk', 'blueprint']) assert.ok(THEME_FILES.includes(`${id}.json`), id);
});

test('the default theme maps every colour to itself (existing links look as before)', () => {
  const c = buildColorMap(DEFAULT_THEME, TABLE);
  assert.ok(c.identity);
  for (const row of TABLE.filter((r) => r.role !== 'clay-roof')) assert.equal(c.map.get(rgbKey(row.rgb)), rgbKey(row.rgb), row.name);
  const t = resolveTheme(DEFAULT_THEME);
  assert.equal(themeCss(t).background, 'linear-gradient(180deg, #8fd0ff 0%, #c9ecff 55%, #fff6e4 100%)');
  assert.equal(t.outline.crease_color, '#2d2a32');
});

test('a theme restyles palette, shades, ink and clay through one exact table', () => {
  const theme = readJson('storybook.json');
  const c = buildColorMap(theme, TABLE);
  assert.ok(!c.identity);
  const byName = (n) => c.byName.get(n);
  const snow = TABLE.find((r) => r.name === 'snow');
  assert.notDeepEqual(byName('snow'), snow.rgb);
  assert.deepEqual(byName('snow.shade'), byName('snow').map((v) => Math.round(v * theme.shade.factor)));
  assert.deepEqual(byName('ink'), hexToRgb(theme.ink));
  assert.deepEqual(byName('clay'), hexToRgb(theme.clay.wall));
  assert.deepEqual(byName('clay-hatch'), hexToRgb(theme.clay.hatch));
  // explicit colours win over rules; a single shade colour replaces the derived ones
  const t = clone(theme);
  t.palette.colors = { snow: '#123456' };
  t.shade = { color: '#ffffff' };
  const c2 = buildColorMap(t, TABLE);
  assert.deepEqual(c2.byName.get('snow'), [0x12, 0x34, 0x56]);
  assert.deepEqual(c2.byName.get('brick.shade'), [255, 255, 255]);
});

test('palette rules: ramp, hue, saturation, lightness and mix', () => {
  assert.deepEqual(applyRule([0, 0, 0], { ramp: ['#ff0000', '#0000ff'] }), [255, 0, 0]);
  assert.deepEqual(applyRule([255, 255, 255], { ramp: ['#ff0000', '#00ff00', '#0000ff'] }), [0, 0, 255]);
  const grey = applyRule([200, 60, 40], { saturation: 0 });
  assert.ok(grey[0] === grey[1] && grey[1] === grey[2], String(grey));
  assert.deepEqual(applyRule([10, 20, 30], { mix: { color: '#ffffff', amount: 1 } }), [255, 255, 255]);
  assert.deepEqual(applyRule([128, 128, 128], { lightness: 100 }), [255, 255, 255]);
  const turned = applyRule([255, 0, 0], { hue: 120 });
  assert.ok(turned[1] > 250 && turned[0] < 5 && turned[2] < 5, String(turned));
});

test('the honesty invariant: clay may never pass for a photographed surface', () => {
  const base = readJson('storybook.json');
  // clay painted in a colour the theme gives to observed stone
  let t = clone(base);
  t.clay.wall = rgbToHex(buildColorMap(base, TABLE).byName.get('stone-light'));
  let v = validateTheme(t, TABLE);
  assert.ok(!v.ok && v.errors.some((e) => /honesty: clay\.wall .*stone-light/.test(e)), v.errors.join('; '));
  // or the palette pulled onto the clay colour
  t = clone(base);
  t.palette.rules.push({ match: 'neutral', mix: { color: base.clay.wall, amount: 0.97 } });
  v = validateTheme(t, TABLE);
  assert.ok(!v.ok && v.errors.some((e) => /^honesty:/.test(e)), v.errors.join('; '));
  // an invisible hatch, in the textures or on untextured faces
  t = clone(base);
  t.clay.hatch = t.clay.wall;
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /hatch invisible/.test(e)));
  t = clone(base);
  t.clay.hatch_shade = 0.99;
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /hatch_shade/.test(e)));
  // a warm light that turns clay into a sand-coloured wall is caught too, although the colours as
  // specified are far apart
  t = readJson('dusk.json');                                     // clay 15.1 from every observed colour as specified ...
  t.gradient = null;
  t.lights.sun = { color: '#40a0ff', intensity: 4 };
  t.lights.hemisphere.sky = '#80c0ff';
  v = validateTheme(t, TABLE);                                   // ... 1.6 from cream under a blue light
  assert.ok(!v.ok && v.errors.some((e) => /^honesty: .*cream.*光照/.test(e)), v.errors.join('; '));
  t.lights.sun.color = '#ffffff';
  t.lights.hemisphere.sky = '#ffffff';
  assert.ok(validateTheme(t, TABLE).ok, validateTheme(t, TABLE).errors.join('; '));
  // the default look is at the edge it is allowed to be at (7.9 from snow, 7.6 in its own light), and honest
  const d = validateTheme(DEFAULT_THEME, TABLE);
  assert.ok(d.ok && d.minClayDeltaE >= MIN_CLAY_DELTA_E && d.minClayDeltaE < 8.5, String(d.minClayDeltaE));
  assert.ok(deltaE00Rgb(hexToRgb('#e4e8ee'), [250, 248, 240]) > MIN_CLAY_DELTA_E);
});

test('the schema: every palette entry mapped, no typos, Chinese name and description', () => {
  const base = readJson('dusk.json');
  let t = clone(base);
  t.palette.rules = [{ match: 'neutral' }];
  let v = validateTheme(t, TABLE);
  assert.ok(!v.ok && v.errors.some((e) => /not mapped: .*brick/.test(e) && !/snow/.test(e)), v.errors.join('; '));
  t = clone(base);
  t.palette.rules = [{ match: 'marble' }, { match: '*' }];
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /rules\[0\]\.match/.test(e)));
  t = clone(base);
  t.outlines = null;
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /unknown key "outlines"/.test(e)));
  t = clone(base);
  t.ink = 'purple';
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /^ink:/.test(e)));
  t = clone(base);
  t.name = 'Dusk';
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /^name:/.test(e)));
  t = clone(base);
  t.description = '第一行\n第二行';
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /^description:/.test(e)));
  t = clone(base);
  t.toon.steps = [0.4, 0.8];
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /toon\.steps/.test(e)));
  t = clone(base);
  t.background = { type: 'gradient', stops: [{ color: '#000000', at: 0.5 }, { color: '#ffffff', at: 0.2 }] };
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /background\.stops/.test(e)));
  t = clone(base);
  t.palette.colors = { marble: '#ffffff' };
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /"marble" is not a palette name/.test(e)));
  assert.ok(!validateTheme(null, TABLE).ok && !validateTheme([], TABLE).ok);
});

test('remapping atlas pixels: exact colours, near misses, and colours not in the table', () => {
  const c = buildColorMap(readJson('blueprint.json'), TABLE);
  const snow = TABLE.find((r) => r.name === 'snow').rgb;
  const ink = TABLE.find((r) => r.name === 'ink').rgb;
  const px = [snow, snow, ink, [snow[0] - 1, snow[1], snow[2] + 1], [1, 254, 3]];
  const data = new Uint8ClampedArray(px.flatMap((p) => [...p, 255]));
  const stats = remapPixels(data, c.map, c.keys);
  assert.deepEqual(stats, { exact: 3, near: 1, unknown: 1 });
  const out = [0, 1, 2, 3, 4].map((i) => [...data.slice(4 * i, 4 * i + 3)]);
  assert.deepEqual(out[0], c.byName.get('snow'));
  assert.deepEqual(out[2], c.byName.get('ink'));
  assert.deepEqual(out[3], c.byName.get('snow'));
  assert.deepEqual(out[4], [1, 254, 3]); // left as it is
  assert.ok([...data].filter((_, i) => i % 4 === 3).every((a) => a === 255));
});

test('loading a theme: the default needs no request, a bad id or file falls back to the default', async () => {
  const calls = [];
  const files = { 'themes/dusk.json': readJson('dusk.json'), 'themes/odd.json': { id: 'other' } };
  const fetchJson = async (url) => {
    calls.push(url);
    if (!(url in files)) throw new Error('HTTP 404');
    return files[url];
  };
  let r = await loadTheme('default', fetchJson);
  assert.equal(r.theme, DEFAULT_THEME);
  assert.equal(r.error, null);
  assert.deepEqual(calls, []);
  r = await loadTheme('dusk', fetchJson);
  assert.equal(r.theme.id, 'dusk');
  assert.equal(r.error, null);
  assert.deepEqual(calls, ['themes/dusk.json']);
  r = await loadTheme('../secret', fetchJson);
  assert.equal(r.theme, DEFAULT_THEME);
  assert.match(r.error, /无效/);
  r = await loadTheme('missing', fetchJson);
  assert.equal(r.theme, DEFAULT_THEME);
  assert.match(r.error, /404/);
  r = await loadTheme('odd', fetchJson);
  assert.equal(r.theme, DEFAULT_THEME);
  assert.match(r.error, /id/);
});
