// Run from the repo root: node --test 'web/tests/*.test.mjs'
// Viewer themes (web/themes.js, web/themes/*.json): schema, honesty invariant, colour remapping, loading.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  DEFAULT_THEME,
  MIN_CLAY_DELTA_E,
  MIN_HATCH_DELTA_E,
  applyRule,
  buildColorMap,
  deltaE00,
  deltaE00Rgb,
  hexToRgb,
  lightConditions,
  loadTheme,
  remapPixels,
  resolveTheme,
  rgbKey,
  rgbToHex,
  rgbToLab,
  scaleLinear,
  srgbToLinear,
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

test('the default theme keeps every observed colour as it is and only moves the clay', () => {
  const c = buildColorMap(DEFAULT_THEME, TABLE);
  for (const row of TABLE.filter((r) => ['palette', 'shade', 'ink'].includes(r.role))) {
    assert.equal(c.map.get(rgbKey(row.rgb)), rgbKey(row.rgb), row.name);
  }
  // the clay texels of the atlases take the default theme's clay: a clay-only change is a restyle
  assert.ok(!c.identity);
  assert.deepEqual(c.byName.get('clay'), hexToRgb(DEFAULT_THEME.clay.wall));
  assert.deepEqual(c.byName.get('clay-hatch'), hexToRgb(DEFAULT_THEME.clay.hatch));
  assert.notDeepEqual(hexToRgb(DEFAULT_THEME.clay.wall), TABLE.find((r) => r.role === 'clay').rgb);
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
  const mapped = buildColorMap(base, TABLE).byName;
  const rejects = (t, re) => {
    const v = validateTheme(t, TABLE);
    assert.ok(!v.ok && v.errors.some((e) => re.test(e)), `${re}: ${v.errors.join('; ') || 'accepted'}`);
  };
  // the threshold itself, and themes on either side of it
  assert.equal(MIN_CLAY_DELTA_E, 7.5);
  assert.equal(MIN_HATCH_DELTA_E, 3.0);
  let t = clone(DEFAULT_THEME);
  t.clay = { ...t.clay, wall: '#e8e2ee', hatch: '#d2c8dc' };                   // 6.7 from stone-light in a cast shadow
  rejects(t, /^honesty: clay\.wall is only ΔE00 6\.7 from stone-light/);
  t = clone(base);
  t.clay.wall = '#f2e5ea';
  t.clay.hatch = '#c0a8b8';                                                   // 5.1
  rejects(t, /^honesty: clay\.wall is only ΔE00 5\.1 /);
  // every clay colour the viewer can show is checked: each of these is exactly an observed colour as mapped
  t = clone(base);
  t.clay.wall = rgbToHex(mapped.get('stone-light'));
  rejects(t, /^honesty: clay\.wall is only ΔE00 0\.0 from stone-light as this theme maps it \(如设定\)/);
  t = clone(base);
  t.clay.hatch = rgbToHex(mapped.get('peach'));
  rejects(t, /^honesty: clay\.hatch is only ΔE00 0\.0 from peach/);
  t = clone(base);
  t.clay.roof = rgbToHex(mapped.get('mint'));
  rejects(t, /^honesty: clay\.roof is only ΔE00 0\.0 from mint/);
  t = clone(DEFAULT_THEME);
  t.clay = { ...t.clay, face: '#f0d6a0', hatch_shade: 0.7 };                  // whole locked walls in `sand`
  rejects(t, /^honesty: clay\.face is only ΔE00 0\.0 from sand/);
  t = clone(DEFAULT_THEME);
  t.clay = { ...t.clay, face: '#72e682', hatch_shade: 0.7 };                  // their stripes in `leaf`
  rejects(t, /^honesty: clay\.face hatch is only ΔE00 0\.0 from leaf/);
  // ... the stripes on every locked roof (clay.roof x hatch_shade) included
  t = clone(base);
  t.clay.roof = '#c1fcc2';
  t.clay.hatch_shade = 0.6;
  rejects(t, /^honesty: clay\.roof hatch is only ΔE00 0\.0 from leaf/);
  t = readJson('dusk.json');
  t.clay.roof = '#fcd184';
  t.clay.hatch_shade = 0.7;
  rejects(t, /^honesty: clay\.roof hatch is only ΔE00 0\.0 from sunflower/);
  // ... against every colour an observed facade can show: shades and ink too
  t = clone(base);
  t.clay.roof = rgbToHex(mapped.get('sky.shade'));
  rejects(t, /^honesty: clay\.roof is only ΔE00 0\.0 from sky\.shade/);
  t = clone(base);
  t.ink = base.clay.wall;
  rejects(t, /^honesty: clay\.wall is only ΔE00 0\.0 from ink/);
  // or the palette pulled onto the clay colour
  t = clone(base);
  t.palette.rules.push({ match: 'neutral', mix: { color: base.clay.wall, amount: 0.97 } });
  rejects(t, /^honesty:/);
});

test('the honesty invariant holds in the light the viewer renders with', () => {
  const base = readJson('storybook.json');
  const rejects = (t, re) => {
    const v = validateTheme(t, TABLE);
    assert.ok(!v.ok && v.errors.some((e) => re.test(e)), `${re}: ${v.errors.join('; ') || 'accepted'}`);
  };
  // a coloured light that turns clay into a sand-coloured wall, although the colours as specified are far apart
  let t = readJson('dusk.json');
  t.gradient = null;
  t.lights.sun = { color: '#40a0ff', intensity: 4 };
  t.lights.hemisphere.sky = '#80c0ff';
  rejects(t, /^honesty: clay\.hatch is only ΔE00 1\.9 from cream .*墙面·明暗档 1\)/);
  // the hemisphere's ground colour lights every wall half and half with the sky
  t = clone(base);
  t.lights.hemisphere.ground = '#00e000';
  rejects(t, /^honesty: clay\.face hatch is only ΔE00 3\.7 from stone .*墙面·投影里\)/);
  t = clone(base);
  t.lights = { hemisphere: { sky: '#ffffff', ground: '#e000e0', intensity: 3 }, sun: { color: '#ffffff', intensity: 0 } };
  rejects(t, /^honesty: clay\.wall is only ΔE00 3\.4 from stone-light/);
  // absolute intensities: a light so bright that walls clip to white, or so dim that clay and stone meet
  t = clone(base);
  t.lights.hemisphere.intensity = 4;
  t.lights.sun.intensity = 5;
  rejects(t, /^honesty: clay\.wall is only ΔE00 0\.0 from snow/);
  t = clone(base);
  t.lights.hemisphere = { ...t.lights.hemisphere, ground: '#33ff00', intensity: 0.5 };
  t.lights.sun.intensity = 0.3;
  rejects(t, /^honesty: clay\.face hatch is only ΔE00 2\.4 from stone/);
  // a wall in a cast shadow (hemisphere light only): the old clay of storybook passes everywhere else
  t = clone(base);
  t.clay = { ...t.clay, wall: '#ecdbe4', roof: '#e6d3de' };
  rejects(t, /^honesty: clay\.face hatch is only ΔE00 7\.4 from stone .*墙面·投影里\)/);
  // ... and the old default clay came within 5.2 of `snow` there (why the default clay moved)
  t = clone(DEFAULT_THEME);
  t.clay = { ...t.clay, wall: '#e4e8ee', hatch: '#d0d7e2', face: '#c6ceda', roof: '#c0c7d3' };
  rejects(t, /^honesty: clay\.wall is only ΔE00 5\.2 from snow .*墙面·投影里\)/);
  // the gradient tint, at the bottom of the building
  t = readJson('dusk.json');
  t.lights.sun.color = '#ffffff';
  t.lights.hemisphere.sky = '#ffffff';
  assert.ok(validateTheme({ ...t, gradient: null }, TABLE).ok);
  t.gradient = { top: '#ffffff', bottom: '#40a0ff', strength: 1 };
  rejects(t, /^honesty: clay\.hatch is only ΔE00 0\.9 from cream .*渐变底部\)/);
  // the default look keeps its clay at 8.0 from `snow` in its darkest light, and is honest
  const d = validateTheme(DEFAULT_THEME, TABLE);
  assert.ok(d.ok && d.minClayDeltaE >= MIN_CLAY_DELTA_E && d.minClayDeltaE < 8.5, String(d.minClayDeltaE));
  assert.deepEqual(d.closest, ['clay.wall', 'snow', '墙面·投影里']);
  // the conditions: walls, slopes and flat roofs, in a cast shadow and at every toon step, absolute
  const conds = lightConditions(DEFAULT_THEME);
  assert.equal(conds.length, 3 * 4);
  const lit = Object.fromEntries(conds);
  const lin = (hex) => hexToRgb(hex).map((c) => srgbToLinear(c / 255));
  const [sky, ground, sun] = [lin('#fdfbff'), lin('#d5e6c4'), lin('#fffaf0')];
  for (let i = 0; i < 3; i += 1) {
    near(lit['墙面·投影里'][i], (1.25 * (sky[i] + ground[i])) / 2 / Math.PI, 1e-9);
    near(lit['平屋顶·明暗档 0.78'][i], (1.25 * sky[i] + 2.0 * sun[i] * 0.78) / Math.PI, 1e-9);
    near(lit['斜屋顶·投影里'][i], (1.25 * (ground[i] + (sky[i] - ground[i]) * 0.75)) / Math.PI, 1e-9);
  }
  // with a gradient: at its bottom, middle and top
  const dusk = readJson('dusk.json');
  const dl = Object.fromEntries(lightConditions(dusk));
  assert.equal(Object.keys(dl).length, 3 * 5 * 3);
  const mid = lin(dusk.gradient.bottom).map((b, i) => 1 + ((b + lin(dusk.gradient.top)[i]) / 2 - 1) * dusk.gradient.strength);
  const wallShadow = lin(dusk.lights.hemisphere.sky).map((v, i) => (1.15 * (v + lin(dusk.lights.hemisphere.ground)[i])) / 2 / Math.PI);
  for (let i = 0; i < 3; i += 1) near(dl['墙面·投影里·渐变中部'][i], wallShadow[i] * mid[i], 1e-9);
});

test('every hatch stays visible in the light the viewer renders with', () => {
  const base = readJson('storybook.json');
  const rejects = (t, re) => {
    const v = validateTheme(t, TABLE);
    assert.ok(!v.ok && v.errors.some((e) => re.test(e)), `${re}: ${v.errors.join('; ') || 'accepted'}`);
  };
  let t = clone(base);
  t.clay.hatch = t.clay.wall;
  rejects(t, /^honesty: clay\.hatch is only ΔE00 0\.0 from clay\.wall .*hatch invisible/);
  t = clone(base);
  t.clay.hatch = '#e4d0db';                                                   // 2.9 in a cast shadow
  rejects(t, /^honesty: clay\.hatch is only ΔE00 2\.9 from clay\.wall .*hatch invisible/);
  t = clone(base);
  t.clay.hatch_shade = 0.99;                                                  // out of range
  assert.ok(validateTheme(t, TABLE).errors.some((e) => /hatch_shade/.test(e)));
  // the stripes on locked roofs: a near-black roof hides them, although it is far from every observed colour
  t = clone(base);
  t.clay.roof = '#0a0a0a';
  const v = validateTheme(t, TABLE);
  assert.ok(v.minClayDeltaE >= MIN_CLAY_DELTA_E, String(v.minClayDeltaE));
  rejects(t, /^honesty: clay\.roof hatch is only ΔE00 0\.2 from clay\.roof .*hatch invisible/);
  // visible as specified, not in the shadow at the foot of a dusk building
  t = readJson('dusk.json');
  t.clay.hatch_shade = 0.85;
  assert.ok(deltaE00Rgb(hexToRgb(t.clay.wall), scaleLinear(hexToRgb(t.clay.wall), 0.85)) >= MIN_HATCH_DELTA_E);
  rejects(t, /^honesty: clay\.face hatch is only ΔE00 2\.3 from clay\.face \(墙面·投影里·渐变底部\): hatch invisible/);
  for (const file of THEME_FILES) assert.ok(validateTheme(readJson(file), TABLE).minHatchDeltaE >= MIN_HATCH_DELTA_E, file);
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
