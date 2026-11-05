// Run from the repo root: node --test 'web/tests/*.test.mjs'
// Viewer themes (web/themes.js, web/themes/*.json): schema, hatch visibility in the viewer's light, loading.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { test } from 'node:test';

import { STRINGS } from '../strings.js';
import {
  DEFAULT_THEME,
  EN_NAME_MAX,
  MIN_HATCH_DELTA_E,
  deltaE00,
  deltaE00Rgb,
  hexToRgb,
  lightConditions,
  loadTheme,
  resolveTheme,
  rgbToHex,
  rgbToLab,
  scaleLinear,
  srgbToLinear,
  themeCss,
  validateTheme,
} from '../themes.js';

const DIR = new URL('../themes/', import.meta.url);
const readJson = (name) => JSON.parse(readFileSync(new URL(name, DIR), 'utf8'));
const THEME_FILES = readdirSync(DIR).filter((f) => f.endsWith('.json') && f !== 'index.json');
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

test('every theme in web/themes is valid and listed in the picker index', () => {
  assert.ok(THEME_FILES.length >= 4, THEME_FILES.join(', '));
  for (const file of THEME_FILES) {
    const theme = readJson(file);
    assert.equal(`${theme.id}.json`, file);
    const v = validateTheme(theme);
    assert.ok(v.ok, `${file}: ${v.errors.join('; ')}`);
    assert.ok(hexToRgb(theme.ink), `${file}: ink`); // the line colour of overlays drawn on the model
  }
  assert.deepEqual([...readJson('index.json').themes].sort(), THEME_FILES.map((f) => f.replace('.json', '')).sort());
  assert.equal(readJson('index.json').themes[0], 'default');
  assert.deepEqual(readJson('default.json'), clone(DEFAULT_THEME));
  for (const id of ['storybook', 'dusk', 'blueprint']) assert.ok(THEME_FILES.includes(`${id}.json`), id);
});

test('the default theme: the clay every clay model showed before D29, the sky, the ink', () => {
  const t = resolveTheme(DEFAULT_THEME);
  assert.deepEqual(t.clay, { wall: '#c8d1df', roof: '#c9d2e2', hatch_shade: 0.84 });
  assert.equal(t.ink, '#2d2a32');
  assert.deepEqual(themeCss(t), { background: 'linear-gradient(180deg, #8fd0ff 0%, #c9ecff 55%, #fff6e4 100%)' });
  assert.equal(t.outline.crease_color, '#2d2a32');
  assert.equal(t.outline.silhouette_color, '#070608');
  // a theme without clay.roof: the wall clay a little darker
  const noRoof = resolveTheme({ ...readJson('storybook.json'), clay: { wall: '#f1dde7' } });
  assert.equal(noRoof.clay.roof, rgbToHex(scaleLinear(hexToRgb('#f1dde7'), 0.93)));
  assert.equal(noRoof.clay.hatch_shade, 0.84);
});

test('the light conditions the hatch is checked under: walls, slopes and flat roofs, shadow and toon steps, gradient', () => {
  const conds = lightConditions(DEFAULT_THEME);
  assert.equal(conds.length, 3 * 4);
  const lit = Object.fromEntries(conds);
  const lin = (hex) => hexToRgb(hex).map((c) => srgbToLinear(c / 255));
  const [sky, ground, sun] = [lin('#fdfbff'), lin('#d5e6c4'), lin('#fffaf0')];
  for (let i = 0; i < 3; i += 1) {
    near(lit['wall·in shadow'][i], (1.25 * (sky[i] + ground[i])) / 2 / Math.PI, 1e-9);
    near(lit['flat roof·toon step 0.78'][i], (1.25 * sky[i] + 2.0 * sun[i] * 0.78) / Math.PI, 1e-9);
    near(lit['roof slope·in shadow'][i], (1.25 * (ground[i] + (sky[i] - ground[i]) * 0.75)) / Math.PI, 1e-9);
  }
  // with a gradient: at its bottom, middle and top
  const dusk = readJson('dusk.json');
  const dl = Object.fromEntries(lightConditions(dusk));
  assert.equal(Object.keys(dl).length, 3 * 5 * 3);
  const mid = lin(dusk.gradient.bottom).map((b, i) => 1 + ((b + lin(dusk.gradient.top)[i]) / 2 - 1) * dusk.gradient.strength);
  const wallShadow = lin(dusk.lights.hemisphere.sky).map((v, i) => (1.15 * (v + lin(dusk.lights.hemisphere.ground)[i])) / 2 / Math.PI);
  for (let i = 0; i < 3; i += 1) near(dl['wall·in shadow·gradient middle'][i], wallShadow[i] * mid[i], 1e-9);
});

test('every hatch stays visible in the light the viewer renders with', () => {
  const base = readJson('storybook.json');
  const rejects = (t, re) => {
    const v = validateTheme(t);
    assert.ok(!v.ok && v.errors.some((e) => re.test(e)), `${re}: ${v.errors.join('; ') || 'accepted'}`);
  };
  assert.equal(MIN_HATCH_DELTA_E, 3.0);
  let t = clone(base);
  t.clay.hatch_shade = 0.99;                                                  // out of range
  assert.ok(validateTheme(t).errors.some((e) => /hatch_shade/.test(e)));
  // a near-black wall or roof hides its stripes
  t = clone(base);
  t.clay.roof = '#0a0a0a';
  rejects(t, /^hatch: clay\.roof hatch is only ΔE00 0\.2 from clay\.roof .*hatch invisible/);
  t = clone(base);
  t.clay = { wall: '#0a0a0a' };                                               // (the roof follows the wall)
  rejects(t, /^hatch: clay\.wall hatch is only ΔE00 0\.2 from clay\.wall .*hatch invisible/);
  rejects(t, /^hatch: clay\.roof hatch /);
  // a light so bright that both clip to white
  t = clone(base);
  t.lights.hemisphere.intensity = 4;
  t.lights.sun.intensity = 5;
  rejects(t, /^hatch: clay\.wall hatch is only ΔE00 0\.0 from clay\.wall \(wall·toon step/);
  // visible as specified, not in the shadow at the foot of a dusk building
  t = readJson('dusk.json');
  t.clay.hatch_shade = 0.85;
  assert.ok(deltaE00Rgb(hexToRgb(t.clay.wall), scaleLinear(hexToRgb(t.clay.wall), 0.85)) >= MIN_HATCH_DELTA_E);
  rejects(t, /^hatch: clay\.wall hatch is only ΔE00 2\.3 from clay\.wall \(wall·in shadow·gradient bottom\): hatch invisible/);
  for (const file of THEME_FILES) assert.ok(validateTheme(readJson(file)).minHatchDeltaE >= MIN_HATCH_DELTA_E, file);
  assert.ok(validateTheme(DEFAULT_THEME).minHatchDeltaE >= MIN_HATCH_DELTA_E);
});

test('the schema: no typos, no keys of the old colour themes, Chinese name and description', () => {
  const base = readJson('dusk.json');
  let t = clone(base);
  t.outlines = null;
  assert.ok(validateTheme(t).errors.some((e) => /unknown key "outlines"/.test(e)));
  // D29: models are never coloured, so a theme has no palette, shade, frame or atlas clay any more
  for (const key of ['palette', 'shade', 'locked_frame']) {
    t = clone(base);
    t[key] = key === 'locked_frame' ? '#ffffff' : {};
    assert.ok(validateTheme(t).errors.some((e) => e === `unknown key "${key}"`), key);
  }
  for (const key of ['hatch', 'face']) {
    t = clone(base);
    t.clay[key] = '#ffffff';
    assert.ok(validateTheme(t).errors.some((e) => e === `clay: unknown key "${key}"`), key);
  }
  t = clone(base);
  t.ink = 'purple';
  assert.ok(validateTheme(t).errors.some((e) => /^ink:/.test(e)));
  t = clone(base);
  delete t.ink;
  assert.ok(validateTheme(t).errors.some((e) => /^ink:/.test(e)));
  t = clone(base);
  t.clay.wall = '#fff';
  assert.ok(validateTheme(t).errors.some((e) => /^clay\.wall:/.test(e)));
  t = clone(base);
  t.name = 'Dusk';
  assert.ok(validateTheme(t).errors.some((e) => /^name:/.test(e)));
  t = clone(base);
  t.description = '第一行\n第二行';
  assert.ok(validateTheme(t).errors.some((e) => /^description:/.test(e)));
  t = clone(base);
  t.toon.steps = [0.4, 0.8];
  assert.ok(validateTheme(t).errors.some((e) => /toon\.steps/.test(e)));
  t = clone(base);
  t.background = { type: 'gradient', stops: [{ color: '#000000', at: 0.5 }, { color: '#ffffff', at: 0.2 }] };
  assert.ok(validateTheme(t).errors.some((e) => /background\.stops/.test(e)));
  assert.ok(!validateTheme(null).ok && !validateTheme([]).ok);
});

test('the schema: every theme also has an English name and description (D32), one line each, no Chinese', () => {
  for (const file of THEME_FILES) {
    const theme = readJson(file);
    assert.deepEqual(Object.keys(theme.en).sort(), ['description', 'name'], file);
    assert.ok(theme.en.name.length <= EN_NAME_MAX, `${file}: en.name fits the picker`);
    assert.deepEqual(validateTheme(theme).warnings, [], file);
  }
  // the default theme's names are web/strings.js themes.default.* (DEFAULT_THEME = default.json)
  assert.deepEqual(DEFAULT_THEME.en, { name: STRINGS['themes.default.name'].en, description: STRINGS['themes.default.description'].en });
  assert.equal(DEFAULT_THEME.name, STRINGS['themes.default.name'].zh);
  const base = readJson('dusk.json');
  const errors = (en) => {
    const t = clone(base);
    if (en === undefined) delete t.en;
    else t.en = en;
    return validateTheme(t).errors;
  };
  assert.ok(errors(undefined).some((e) => /^en: /.test(e)), 'required');
  assert.ok(errors('Dusk').some((e) => /^en: /.test(e)));
  assert.ok(errors({ name: '黄昏渐变', description: 'Dusk.' }).some((e) => /^en\.name: one line of English text/.test(e)));
  assert.ok(errors({ name: 'Dusk（傍晚）', description: 'Dusk.' }).some((e) => /^en\.name:/.test(e)), 'full-width punctuation');
  assert.ok(errors({ name: ' ', description: 'Dusk.' }).some((e) => /^en\.name:/.test(e)));
  assert.ok(errors({ name: 'Dusk', description: 'Two\nlines' }).some((e) => /^en\.description:/.test(e)));
  assert.ok(errors({ name: 'Dusk' }).some((e) => /^en\.description:/.test(e)));
  assert.ok(errors({ name: 'Dusk', description: 'Dusk.', note: 'x' }).some((e) => e === 'en: unknown key "note"'));
  const long = clone(base);
  long.en.name = 'A rather long English style name';
  const v = validateTheme(long);
  assert.ok(v.ok);
  assert.deepEqual(v.warnings, ['en.name: keep it short (the picker shows it)']);
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
