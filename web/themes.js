// Themes of the stylized-model viewer: how a published model looks, kept apart from what was measured.
//
// A model (crowd3d/stylize.py) is a clay model: the viewer paints every building face by its kind (roofs in
// clay.roof, everything else in clay.wall) with a faint hatch, whatever colours or textures a GLB carries. A
// theme (web/themes/<id>.json, schema in docs/ART_DIRECTION.md, decision D7) sets that look: clay, hatch,
// outlines, toon steps, lights, background, ground and an optional vertical gradient, plus ink, the colour of
// marks drawn over the model (such as the photo-pose layer). It never changes the shape or the provenance a
// model shows.
//
// validateTheme() checks the schema (a typo must not silently fall back) and that the hatch stays visible
// on walls and roofs in the light the viewer actually renders with (lightConditions(): hemisphere sky and
// ground, sun, toon steps, cast shadows, gradient, absolute intensities).
//
// A theme is independent of the page language and of light / dark mode (D32): it carries its name and description in
// Chinese (name, description) and in English (en: {name, description}); the picker shows the one of the page language.
//
// Pure functions only (no DOM, no three.js): the node tests import this module directly.

import { lang, t } from './i18n.js';
import { STRINGS } from './strings.js';

export const THEME_SCHEMA = 1;
export const THEME_ID_RE = /^[a-z0-9][a-z0-9-]{0,39}$/;
export const MIN_HATCH_DELTA_E = 3.0; // clay vs its hatch stripes, on walls and roofs, in every light
export const HATCH_SHADE_RANGE = [0.5, 0.92]; // hatch stripes = clay colour x this (linear)
export const DEFAULT_ROOF_SHADE = 0.93; // roofs of a theme without clay.roof: clay.wall x this (linear)

// The original look of the viewer (themes/default.json is this object; a test keeps them equal). The
// silhouette is #070608, not ink #2d2a32: the old viewer converted that colour to linear twice, and the look
// was kept. The clay is the colour every clay wall showed before D29 (#c8d1df), so models look the same. Its names
// in both languages are web/strings.js themes.default.* (data of the theme, the same whatever the page language).
export const DEFAULT_THEME = Object.freeze({
  id: 'default',
  schema: 1,
  name: STRINGS['themes.default.name'].zh,
  description: STRINGS['themes.default.description'].zh,
  en: { name: STRINGS['themes.default.name'].en, description: STRINGS['themes.default.description'].en },
  ink: '#2d2a32',
  clay: { wall: '#c8d1df', roof: '#c9d2e2', hatch_shade: 0.84 },
  outline: { color: '#2d2a32', silhouette_color: '#070608', width_px: 2.5, crease_width_px: 1.75 },
  toon: { steps: [0.5, 0.78, 1.0] },
  lights: {
    hemisphere: { sky: '#fdfbff', ground: '#d5e6c4', intensity: 1.25 },
    sun: { color: '#fffaf0', intensity: 2.0 },
  },
  background: {
    type: 'gradient',
    stops: [
      { color: '#8fd0ff', at: 0 },
      { color: '#c9ecff', at: 0.55 },
      { color: '#fff6e4', at: 1 },
    ],
  },
  ground: { color: '#cfeab8', rim: '#a9d48c' },
  gradient: null,
});

// ------------------------------------------------------------------ colours

const HEX_RE = /^#[0-9a-fA-F]{6}$/;

export function hexToRgb(hex) {
  if (typeof hex !== 'string' || !HEX_RE.test(hex)) return null;
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

export function rgbToHex(rgb) {
  return '#' + rgb.map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('');
}

export const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
export const linearToSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** An sRGB colour scaled in linear light (what a lit or shaded face does), back in 0..255 sRGB. */
export function scaleLinear(rgb, factor) {
  return rgb.map((v) => Math.round(255 * linearToSrgb(Math.min(1, srgbToLinear(v / 255) * factor))));
}

/** sRGB 0..255 -> CIE L*a*b* (D65). */
export function rgbToLab(rgb) {
  const [r, g, b] = rgb.map((v) => srgbToLinear(v / 255));
  const x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
  const y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  const z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;
  const f = (t) => (t > (6 / 29) ** 3 ? Math.cbrt(t) : t / (3 * (6 / 29) ** 2) + 4 / 29);
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}

/** CIEDE2000 colour difference of two L*a*b* colours. */
export function deltaE00(lab1, lab2) {
  const [L1, a1, b1] = lab1;
  const [L2, a2, b2] = lab2;
  const rad = Math.PI / 180;
  const Cb = (Math.hypot(a1, b1) + Math.hypot(a2, b2)) / 2;
  const G = 0.5 * (1 - Math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)));
  const a1p = (1 + G) * a1;
  const a2p = (1 + G) * a2;
  const C1p = Math.hypot(a1p, b1);
  const C2p = Math.hypot(a2p, b2);
  const hue = (a, b) => (a === 0 && b === 0 ? 0 : ((Math.atan2(b, a) / rad) + 360) % 360);
  const h1p = hue(a1p, b1);
  const h2p = hue(a2p, b2);
  const dLp = L2 - L1;
  const dCp = C2p - C1p;
  let dhp = 0;
  if (C1p * C2p !== 0) {
    dhp = h2p - h1p;
    if (dhp > 180) dhp -= 360;
    else if (dhp < -180) dhp += 360;
  }
  const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * rad);
  const Lbp = (L1 + L2) / 2;
  const Cbp = (C1p + C2p) / 2;
  let hbp = h1p + h2p;
  if (C1p * C2p !== 0) {
    if (Math.abs(h1p - h2p) <= 180) hbp = (h1p + h2p) / 2;
    else hbp = h1p + h2p < 360 ? (h1p + h2p + 360) / 2 : (h1p + h2p - 360) / 2;
  }
  const T = 1 - 0.17 * Math.cos((hbp - 30) * rad) + 0.24 * Math.cos(2 * hbp * rad)
    + 0.32 * Math.cos((3 * hbp + 6) * rad) - 0.2 * Math.cos((4 * hbp - 63) * rad);
  const dTheta = 30 * Math.exp(-(((hbp - 275) / 25) ** 2));
  const Rc = 2 * Math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7));
  const Sl = 1 + (0.015 * (Lbp - 50) ** 2) / Math.sqrt(20 + (Lbp - 50) ** 2);
  const Sc = 1 + 0.045 * Cbp;
  const Sh = 1 + 0.015 * Cbp * T;
  const Rt = -Math.sin(2 * dTheta * rad) * Rc;
  return Math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh));
}

export const deltaE00Rgb = (p, q) => deltaE00(rgbToLab(p), rgbToLab(q));

/** Theme with every optional field filled in (what the viewer draws with). */
export function resolveTheme(theme) {
  const t = JSON.parse(JSON.stringify(theme));
  t.clay = { hatch_shade: 0.84, ...t.clay };
  t.clay.roof = t.clay.roof ?? rgbToHex(scaleLinear(hexToRgb(t.clay.wall), DEFAULT_ROOF_SHADE));
  if (t.outline) {
    t.outline = {
      silhouette_color: t.outline.color,
      crease_color: t.outline.color,
      crease_width_px: Math.max(1, 0.7 * t.outline.width_px),
      ...t.outline,
    };
  }
  t.gradient = t.gradient ?? null;
  return t;
}

// ------------------------------------------------------------------ validation

const CJK_RE = /[㐀-鿿]/;
// any Chinese character or CJK / full-width punctuation: none of it belongs in the English name or description
const ANY_CJK_RE = /[\p{Script=Han}\u3000-\u303f\uff00-\uffef]/u;
const TOP_KEYS = new Set(['id', 'schema', 'name', 'description', 'en', 'ink', 'clay', 'outline', 'toon', 'lights',
  'background', 'ground', 'gradient']);
export const NAME_MAX = 12; // characters of the Chinese name the picker shows comfortably
export const EN_NAME_MAX = 24; // the same for the English name

// Faces the light conditions sample, by the up component of their normal: walls, steep roof slopes, flat roofs. The
// conditions' names are technical text for the validator's messages (developer-facing, not translated).
export const LIGHT_SURFACES = [['wall', 0], ['roof slope', 0.5], ['flat roof', 1]];
const GRADIENT_POINTS = [['gradient bottom', 0], ['gradient middle', 0.5], ['gradient top', 1]];

/**
 * The light the viewer (web/stylized.js: three.js r160 MeshToonMaterial, physical light units, no tone
 * mapping) puts on a face of the building, as absolute linear RGB multipliers of the face's colour:
 * (hemisphere + sun x toon step) / PI, times the theme's vertical gradient tint. The hemisphere light mixes
 * ground and sky by the normal (half and half on a wall, sky alone on a flat roof); the sun adds its colour
 * x intensity x one of the toon steps (any step: the step depends on the face's direction), or nothing in a
 * cast shadow. Sampled on walls, steep roof slopes and flat roofs, in the shadow and at every toon step, and
 * with a gradient at its bottom, middle and top. [[label, [r, g, b]], ...]; a colour shows as
 * shade(colour, multiplier) (clamped at 1 like the framebuffer).
 */
export function lightConditions(theme) {
  const t = resolveTheme(theme);
  const lin = (hex) => hexToRgb(hex).map((c) => srgbToLinear(c / 255));
  const sky = lin(t.lights.hemisphere.sky);
  const ground = lin(t.lights.hemisphere.ground);
  const sun = lin(t.lights.sun.color);
  const grads = t.gradient
    ? GRADIENT_POINTS.map(([label, at]) => {
      const top = lin(t.gradient.top);
      const bottom = lin(t.gradient.bottom);
      return [`·${label}`, bottom.map((b, i) => 1 + (b + (top[i] - b) * at - 1) * t.gradient.strength)];
    })
    : [['', [1, 1, 1]]];
  const out = [];
  for (const [surface, up] of LIGHT_SURFACES) {
    const w = 0.5 * up + 0.5;
    const hemi = sky.map((s, i) => t.lights.hemisphere.intensity * (ground[i] + (s - ground[i]) * w));
    for (const step of [0, ...t.toon.steps]) {
      const lit = hemi.map((h, i) => (h + t.lights.sun.intensity * sun[i] * step) / Math.PI);
      for (const [gl, g] of grads) {
        out.push([`${surface}·${step === 0 ? 'in shadow' : `toon step ${step}`}${gl}`, lit.map((v, i) => v * g[i])]);
      }
    }
  }
  return out;
}

/** An sRGB colour (0..255) under a light condition of lightConditions(), as the framebuffer holds it. */
export const shade = (rgb, k) => rgb.map((c, i) => Math.round(255 * linearToSrgb(Math.min(1, srgbToLinear(c / 255) * k[i]))));

/**
 * Check a theme. Returns {ok, errors, warnings, minHatchDeltaE}. Errors: schema (types, ranges, unknown keys:
 * a typo, or a key of the old colour themes such as palette, must not silently fall back) and an invisible
 * hatch: the stripes the viewer draws on walls and roofs (clay x hatch_shade) must stay at least
 * MIN_HATCH_DELTA_E (CIEDE2000) from their clay, as specified and under every light condition of
 * lightConditions().
 */
export function validateTheme(theme) {
  const errors = [];
  const warnings = [];
  const err = (m) => errors.push(m);
  const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
  const num = (v, lo, hi) => typeof v === 'number' && Number.isFinite(v) && v >= lo && v <= hi;
  const hex = (v, where) => {
    if (!hexToRgb(v)) err(`${where}: not a #rrggbb colour (${JSON.stringify(v)})`);
  };
  const keys = (obj, allowed, where) => {
    for (const k of Object.keys(obj)) if (!allowed.includes(k)) err(`${where}: unknown key "${k}"`);
  };
  if (!isObj(theme)) return { ok: false, errors: ['a theme is a JSON object'], warnings, minHatchDeltaE: null };
  for (const k of Object.keys(theme)) if (!TOP_KEYS.has(k)) err(`unknown key "${k}"`);
  if (typeof theme.id !== 'string' || !THEME_ID_RE.test(theme.id)) err('id: lowercase letters, digits and "-" (the file name)');
  if (theme.schema !== THEME_SCHEMA) err(`schema: must be ${THEME_SCHEMA}`);
  for (const k of ['name', 'description']) {
    const v = theme[k];
    if (typeof v !== 'string' || !v.trim() || !CJK_RE.test(v) || /\n/.test(v)) err(`${k}: one line of Chinese text`);
  }
  if (typeof theme.name === 'string' && theme.name.length > NAME_MAX) warnings.push('name: keep it short (the picker shows it)');
  // the same in English, for an English page
  const en = theme.en;
  if (!isObj(en)) err('en: {name, description} in English');
  else {
    keys(en, ['name', 'description'], 'en');
    for (const k of ['name', 'description']) {
      const v = en[k];
      if (typeof v !== 'string' || !v.trim() || ANY_CJK_RE.test(v) || /\n/.test(v)) err(`en.${k}: one line of English text, no Chinese`);
    }
    if (typeof en.name === 'string' && en.name.length > EN_NAME_MAX) warnings.push('en.name: keep it short (the picker shows it)');
  }
  // the look
  hex(theme.ink, 'ink');
  const clay = theme.clay;
  if (!isObj(clay)) err('clay: {wall, roof?, hatch_shade?}');
  else {
    keys(clay, ['wall', 'roof', 'hatch_shade'], 'clay');
    hex(clay.wall, 'clay.wall');
    if (clay.roof !== undefined) hex(clay.roof, 'clay.roof');
    if (clay.hatch_shade !== undefined && !num(clay.hatch_shade, ...HATCH_SHADE_RANGE)) {
      err(`clay.hatch_shade: ${HATCH_SHADE_RANGE.join('..')} (the hatch must stay visible)`);
    }
  }
  if (theme.outline !== null) {
    const o = theme.outline;
    if (!isObj(o)) err('outline: {color, width_px, ...} or null (no outlines)');
    else {
      keys(o, ['color', 'width_px', 'silhouette_color', 'crease_color', 'crease_width_px'], 'outline');
      hex(o.color, 'outline.color');
      for (const k of ['silhouette_color', 'crease_color']) if (o[k] !== undefined) hex(o[k], `outline.${k}`);
      if (!num(o.width_px, 0.5, 8)) err('outline.width_px: 0.5..8');
      if (o.crease_width_px !== undefined && !num(o.crease_width_px, 0.5, 8)) err('outline.crease_width_px: 0.5..8');
    }
  }
  const steps = theme.toon?.steps;
  if (!isObj(theme.toon) || !Array.isArray(steps) || steps.length < 2 || steps.length > 6
    || !steps.every((v, i) => num(v, 0.05, 1) && (i === 0 || v > steps[i - 1])) || steps.at(-1) !== 1) {
    err('toon.steps: 2-6 rising brightness steps in (0, 1], the last one 1');
  }
  const L = theme.lights;
  if (!isObj(L) || !isObj(L.hemisphere) || !isObj(L.sun)) err('lights: {hemisphere: {sky, ground, intensity}, sun: {color, intensity}}');
  else {
    hex(L.hemisphere.sky, 'lights.hemisphere.sky');
    hex(L.hemisphere.ground, 'lights.hemisphere.ground');
    hex(L.sun.color, 'lights.sun.color');
    if (!num(L.hemisphere.intensity, 0, 5)) err('lights.hemisphere.intensity: 0..5');
    if (!num(L.sun.intensity, 0, 6)) err('lights.sun.intensity: 0..6');
  }
  const bg = theme.background;
  if (!isObj(bg) || !['solid', 'gradient'].includes(bg.type)) err('background: {type: "solid", color} or {type: "gradient", stops}');
  else if (bg.type === 'solid') hex(bg.color, 'background.color');
  else if (!Array.isArray(bg.stops) || bg.stops.length < 2 || bg.stops.length > 6
    || !bg.stops.every((s, i) => isObj(s) && hexToRgb(s.color) && num(s.at, 0, 1) && (i === 0 || s.at >= bg.stops[i - 1].at))) {
    err('background.stops: 2-6 {color, at 0..1}, in order');
  }
  if (!isObj(theme.ground)) err('ground: {color, rim}');
  else {
    hex(theme.ground.color, 'ground.color');
    hex(theme.ground.rim, 'ground.rim');
  }
  if (theme.gradient !== undefined && theme.gradient !== null) {
    const g = theme.gradient;
    if (!isObj(g) || !num(g.strength, 0, 1)) err('gradient: {top, bottom, strength 0..1} or null');
    else {
      hex(g.top, 'gradient.top');
      hex(g.bottom, 'gradient.bottom');
    }
  }
  if (errors.length) return { ok: false, errors, warnings, minHatchDeltaE: null };

  // the hatch shows on walls and roofs, in any light of the theme
  const t = resolveTheme(theme);
  const wall = hexToRgb(t.clay.wall);
  const roof = hexToRgb(t.clay.roof);
  const hatches = [
    ['clay.wall hatch', scaleLinear(wall, t.clay.hatch_shade), 'clay.wall', wall],
    ['clay.roof hatch', scaleLinear(roof, t.clay.hatch_shade), 'clay.roof', roof],
  ];
  const hatchMin = hatches.map(() => [Infinity, null]);
  for (const [cond, k] of [['as specified', [1, 1, 1]], ...lightConditions(t)]) {
    hatches.forEach(([, h, , c], i) => {
      const d = deltaE00(rgbToLab(shade(h, k)), rgbToLab(shade(c, k)));
      if (d < hatchMin[i][0]) hatchMin[i] = [d, cond];
    });
  }
  hatches.forEach(([h, , c], i) => {
    const [d, cond] = hatchMin[i];
    if (d < MIN_HATCH_DELTA_E) err(`hatch: ${h} is only ΔE00 ${d.toFixed(1)} from ${c} (${cond}): hatch invisible`);
  });
  const minHatch = Math.min(...hatchMin.map(([d]) => d));
  return { ok: errors.length === 0, errors, warnings, minHatchDeltaE: Math.round(minHatch * 10) / 10 };
}

// ------------------------------------------------------------------ loading

/**
 * A theme's name in the page language (the picker's option, the note about a theme that failed the validator): name in
 * Chinese, en.name in English; null when the theme has no usable one (the caller shows its id instead).
 */
export function themeName(theme) {
  const v = lang() === 'en' ? theme?.en?.name : theme?.name;
  return typeof v === 'string' && v.trim() ? v : null;
}

/** CSS for the page behind the canvas: {background}. */
export function themeCss(theme) {
  const bg = theme.background;
  const background = bg.type === 'solid'
    ? bg.color
    : `linear-gradient(180deg, ${bg.stops.map((s) => `${s.color} ${Math.round(s.at * 100)}%`).join(', ')})`;
  return { background };
}

/**
 * The theme a view asked for, unvalidated (the viewer validates it before drawing):
 * {theme, error}. 'default' needs no request; a missing or malformed file falls back to the default
 * with an error message. fetchJson(url) -> parsed JSON (throws on failure).
 */
export async function loadTheme(id, fetchJson, base = 'themes/') {
  if (!id || id === 'default') return { theme: DEFAULT_THEME, error: null };
  if (!THEME_ID_RE.test(id)) return { theme: DEFAULT_THEME, error: t('themes.error.badId', { id }) };
  try {
    const theme = await fetchJson(`${base}${id}.json`);
    if (!theme || theme.id !== id) return { theme: DEFAULT_THEME, error: t('themes.error.wrongId', { id }) };
    return { theme, error: null };
  } catch (e) {
    return { theme: DEFAULT_THEME, error: t('themes.error.loadFailed', { id, message: e.message ?? e }) };
  }
}
