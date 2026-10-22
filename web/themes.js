// Themes of the stylized-model viewer: how a published model looks, kept apart from what was measured.
//
// A model (crowd3d/stylize.py) paints its facades with a closed set of colours, every one named in the
// GLB's scene extras (style.colors: palette colours with their family, their shades, ink and the clay of
// locked faces), and its atlases are lossless. A theme (web/themes/<id>.json, schema in
// docs/ART_DIRECTION.md) maps those names to new colours and sets the rest of the look (clay, outlines,
// toon steps, lights, background); the viewer remaps the atlases texel by texel through the resulting
// table (an exact colour lookup), so nothing is re-run and the measured structure stays as it was.
//
// Themes may only restyle. validateTheme() refuses a theme under which clay could pass for a photographed
// surface: every clay colour must stay at least MIN_CLAY_DELTA_E (CIEDE2000) from every colour that
// observed facades can show (mapped palette colours, their shades, ink), as specified and under the
// theme's own light and gradient tints, and the hatch must stay visible.
//
// Pure functions only (no DOM, no three.js): the node tests import this module directly.

export const THEME_SCHEMA = 1;
export const THEME_ID_RE = /^[a-z0-9][a-z0-9-]{0,39}$/;
// The current default look keeps its clay at 7.9 from `snow` (7.6 in its own light); docs/STYLIZED_MODEL.md
// §4.2 plans a clay colour at >= 10 from every palette colour (raise this then).
export const MIN_CLAY_DELTA_E = 7.5;
export const MIN_HATCH_DELTA_E = 3.0; // clay vs its hatch stripes (textures and untextured faces)
export const HATCH_SHADE_RANGE = [0.5, 0.92]; // untextured locked faces: hatch = face colour x this (linear)
export const DEFAULT_SHADE_FACTOR = 0.62; // = crowd3d.stylize.SHADE_FACTOR
export const DEFAULT_ROOF_SHADE = 0.93; // untextured locked roofs without clay.roof: face colour x this (linear)

// The look of the viewer before themes existed (themes/default.json is this object; a test keeps them
// equal). Locked faces without a texture show #c6ceda / #c0c7d3 and the silhouette #070608, not the clay
// #e4e8ee / ink #2d2a32 of the textures: the old viewer converted those colours to linear twice. Kept so
// that existing links look exactly as before; other themes can give one clay colour for both.
export const DEFAULT_THEME = Object.freeze({
  id: 'default',
  schema: 1,
  name: '明快扁平',
  description: '明亮的扁平色块配粗墨线描边，三档卡通明暗，现在的默认外观。',
  palette: { rules: [{ match: '*' }] },
  shade: { factor: 0.62 },
  ink: '#2d2a32',
  clay: { wall: '#e4e8ee', hatch: '#d0d7e2', face: '#c6ceda', roof: '#c0c7d3', hatch_shade: 0.84 },
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
  locked_frame: '#8f9aab',
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

export const rgbKey = (rgb) => (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
const keyRgb = (k) => [(k >> 16) & 255, (k >> 8) & 255, k & 255];

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

function rgbToHsl([r, g, b]) {
  const [R, G, B] = [r / 255, g / 255, b / 255];
  const max = Math.max(R, G, B);
  const min = Math.min(R, G, B);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l * 100];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = max === R ? (G - B) / d + (G < B ? 6 : 0) : max === G ? (B - R) / d + 2 : (R - G) / d + 4;
  h *= 60;
  return [h, s * 100, l * 100];
}

function hslToRgb([h, s, l]) {
  const S = Math.max(0, Math.min(100, s)) / 100;
  const L = Math.max(0, Math.min(100, l)) / 100;
  const k = (n) => (n + ((h % 360) + 360) % 360 / 30) % 12;
  const a = S * Math.min(L, 1 - L);
  const f = (n) => L - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0), f(8), f(4)].map((v) => Math.round(v * 255));
}

const mixRgb = (p, q, t) => p.map((v, i) => Math.round(v + (q[i] - v) * t));

// ------------------------------------------------------------------ palette rules

const RULE_KEYS = new Set(['match', 'ramp', 'hue', 'saturation', 'lightness', 'mix']);

/**
 * One rule applied to an sRGB colour, in this order: ramp (the colour's L* 0..100 picks a colour along
 * evenly spaced stops: a gradient map), hue (degrees), saturation (HSL, multiplier), lightness (HSL,
 * points added), mix ({color, amount 0..1}).
 */
export function applyRule(rgb, rule) {
  let out = [...rgb];
  if (Array.isArray(rule.ramp) && rule.ramp.length >= 2) {
    const stops = rule.ramp.map(hexToRgb);
    const t = (Math.max(0, Math.min(100, rgbToLab(out)[0])) / 100) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(t));
    out = mixRgb(stops[i], stops[i + 1], t - i);
  }
  if (rule.hue !== undefined || rule.saturation !== undefined || rule.lightness !== undefined) {
    const [h, s, l] = rgbToHsl(out);
    out = hslToRgb([h + (rule.hue ?? 0), s * (rule.saturation ?? 1), l + (rule.lightness ?? 0)]);
  }
  if (rule.mix) out = mixRgb(out, hexToRgb(rule.mix.color), rule.mix.amount);
  return out;
}

const matchList = (m) => (Array.isArray(m) ? m : [m]);
const ruleMatches = (rule, row) => matchList(rule.match).some((m) => m === '*' || m === row.family || m === row.name);

/** Theme with every optional field filled in (what the viewer draws with). */
export function resolveTheme(theme) {
  const t = JSON.parse(JSON.stringify(theme));
  t.palette = { rules: [], colors: {}, ...t.palette };
  t.shade = t.shade ?? { factor: DEFAULT_SHADE_FACTOR };
  t.clay = { hatch_shade: 0.84, ...t.clay };
  t.clay.face = t.clay.face ?? t.clay.wall;
  t.clay.roof = t.clay.roof ?? rgbToHex(scaleLinear(hexToRgb(t.clay.face), DEFAULT_ROOF_SHADE));
  if (t.outline) {
    t.outline = {
      silhouette_color: t.outline.color,
      crease_color: t.outline.color,
      crease_width_px: Math.max(1, 0.7 * t.outline.width_px),
      ...t.outline,
    };
  }
  t.locked_frame = t.locked_frame ?? DEFAULT_THEME.locked_frame;
  t.gradient = t.gradient ?? null;
  return t;
}

/**
 * The colour each table entry shows under a theme: {byName: Map name -> rgb, map: Map key -> key (old
 * sRGB colour -> new), keys: table colour keys, identity}. table: style.colors of a GLB
 * ([{name, rgb, role, family?, of?}]); shadeFactor: the GLB's style.shade_factor. identity: no colour
 * that atlases or observed faces can hold changes (clay-roof is only ever the flat colour of locked
 * roofs, which the viewer paints with clay.roof anyway), so nothing needs remapping.
 */
export function buildColorMap(theme, table, shadeFactor = DEFAULT_SHADE_FACTOR) {
  const t = resolveTheme(theme);
  const byName = new Map();
  for (const row of table.filter((r) => r.role === 'palette')) {
    let rgb = [...row.rgb];
    for (const rule of t.palette.rules) if (ruleMatches(rule, row)) rgb = applyRule(rgb, rule);
    const fixed = hexToRgb(t.palette.colors?.[row.name]);
    byName.set(row.name, fixed ?? rgb);
  }
  const shadeColor = hexToRgb(t.shade.color);
  const factor = t.shade.factor ?? shadeFactor;
  for (const row of table) {
    if (row.role === 'palette') continue;
    let rgb = null;
    if (row.role === 'shade') {
      const base = byName.get(row.of);
      rgb = shadeColor ?? (base ? base.map((v) => Math.round(v * factor)) : null);
      // the identity theme keeps the exact shade the model was painted with
      if (!shadeColor && base && factor === shadeFactor && rgbKey(base) === rgbKey(table.find((r) => r.name === row.of).rgb)) {
        rgb = [...row.rgb];
      }
    } else if (row.role === 'ink') rgb = hexToRgb(t.ink);
    else if (row.role === 'clay') rgb = hexToRgb(t.clay.wall);
    else if (row.role === 'clay-hatch') rgb = hexToRgb(t.clay.hatch);
    else if (row.role === 'clay-roof') rgb = hexToRgb(t.clay.roof);
    byName.set(row.name, rgb ?? [...row.rgb]);
  }
  const map = new Map();
  let identity = true;
  for (const row of table) {
    const to = rgbKey(byName.get(row.name));
    map.set(rgbKey(row.rgb), to);
    if (to !== rgbKey(row.rgb) && row.role !== 'clay-roof') identity = false;
  }
  return { byName, map, keys: table.map((r) => rgbKey(r.rgb)), identity };
}

/**
 * Remap RGBA pixels in place through map (sRGB key -> key). A colour not in the table (there should be
 * none: atlases are lossless) takes the nearest table colour within `tolerance` (sum of channel
 * differences), else stays as it is. Returns counts {exact, near, unknown} of pixels.
 */
export function remapPixels(data, map, keys, tolerance = 6) {
  const near = new Map();
  const stats = { exact: 0, near: 0, unknown: 0 };
  let lastIn = -1;
  let lastOut = -1;
  let lastKind = 'exact';
  for (let i = 0; i < data.length; i += 4) {
    const k = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    if (k !== lastIn) {
      lastIn = k;
      if (map.has(k)) {
        lastOut = map.get(k);
        lastKind = 'exact';
      } else {
        if (!near.has(k)) {
          const [r, g, b] = keyRgb(k);
          let best = -1;
          let bestD = tolerance + 1;
          for (const q of keys) {
            const [qr, qg, qb] = keyRgb(q);
            const d = Math.abs(qr - r) + Math.abs(qg - g) + Math.abs(qb - b);
            if (d < bestD) {
              bestD = d;
              best = q;
            }
          }
          near.set(k, best);
        }
        const q = near.get(k);
        lastOut = q >= 0 ? map.get(q) : k;
        lastKind = q >= 0 ? 'near' : 'unknown';
      }
    }
    stats[lastKind] += 1;
    data[i] = (lastOut >> 16) & 255;
    data[i + 1] = (lastOut >> 8) & 255;
    data[i + 2] = lastOut & 255;
  }
  return stats;
}

// ------------------------------------------------------------------ validation

const CJK_RE = /[㐀-鿿]/;
const TOP_KEYS = new Set(['id', 'schema', 'name', 'description', 'palette', 'shade', 'ink', 'clay', 'outline', 'toon',
  'lights', 'background', 'ground', 'locked_frame', 'gradient']);

/**
 * The colour casts a theme puts on every face alike, as linear RGB multipliers (max channel 1): its key
 * light (sun and hemisphere sky, weighted by intensity) and, with a vertical gradient, that light times
 * the gradient's top and bottom tints. [[label, [r, g, b]], ...]
 */
export function lightTints(theme) {
  const t = resolveTheme(theme);
  const lin = (hex) => hexToRgb(hex).map((c) => srgbToLinear(c / 255));
  const norm = (v) => v.map((x) => x / Math.max(...v, 1e-9));
  const sun = lin(t.lights.sun.color);
  const sky = lin(t.lights.hemisphere.sky);
  const key = norm(sun.map((v, i) => t.lights.sun.intensity * v + t.lights.hemisphere.intensity * sky[i]));
  const out = [['光照', key]];
  if (t.gradient) {
    for (const [label, end] of [['渐变顶部', 'top'], ['渐变底部', 'bottom']]) {
      const g = lin(t.gradient[end]).map((v) => 1 + (v - 1) * t.gradient.strength);
      out.push([label, norm(key.map((v, i) => v * g[i]))]);
    }
  }
  return out;
}

const tinted = (rgb, k) => rgb.map((c, i) => Math.round(255 * linearToSrgb(Math.min(1, srgbToLinear(c / 255) * k[i]))));

/**
 * Check a theme against a colour table (style.colors). Returns {ok, errors, warnings, minClayDeltaE,
 * closest: [clay entry, palette entry, condition]}. Errors: schema (types, ranges, unknown keys: a typo
 * must not silently fall back), palette entries no rule or colour maps, and the honesty invariant: every
 * clay colour (wall, hatch, untextured face and roof, and the face's hatch stripes) at least
 * MIN_CLAY_DELTA_E (CIEDE2000) from every colour an observed facade can show under the theme (mapped
 * palette colours, their shades, ink), as specified and again under the theme's own light and gradient
 * tints (lightTints(): a warm light must not turn clay into a sand wall either); the hatch at least
 * MIN_HATCH_DELTA_E from its clay.
 */
export function validateTheme(theme, table) {
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
  if (!isObj(theme)) return { ok: false, errors: ['a theme is a JSON object'], warnings, minClayDeltaE: null, closest: null };
  for (const k of Object.keys(theme)) if (!TOP_KEYS.has(k)) err(`unknown key "${k}"`);
  if (typeof theme.id !== 'string' || !THEME_ID_RE.test(theme.id)) err('id: lowercase letters, digits and "-" (the file name)');
  if (theme.schema !== THEME_SCHEMA) err(`schema: must be ${THEME_SCHEMA}`);
  for (const k of ['name', 'description']) {
    const v = theme[k];
    if (typeof v !== 'string' || !v.trim() || !CJK_RE.test(v) || /\n/.test(v)) err(`${k}: one line of Chinese text`);
  }
  if (typeof theme.name === 'string' && theme.name.length > 12) warnings.push('name: keep it short (the picker shows it)');
  // palette
  const palette = theme.palette;
  const families = new Set(table.filter((r) => r.family).map((r) => r.family));
  const names = new Set(table.filter((r) => r.role === 'palette').map((r) => r.name));
  if (!isObj(palette) || !Array.isArray(palette.rules)) err('palette.rules: a list of rules');
  else {
    keys(palette, ['rules', 'colors'], 'palette');
    palette.rules.forEach((rule, i) => {
      const where = `palette.rules[${i}]`;
      if (!isObj(rule)) return err(`${where}: an object`);
      for (const k of Object.keys(rule)) if (!RULE_KEYS.has(k)) err(`${where}: unknown key "${k}"`);
      const m = matchList(rule.match);
      if (!m.length || !m.every((x) => x === '*' || families.has(x) || names.has(x))) {
        err(`${where}.match: "*", a family (${[...families].join(', ')}) or a palette name`);
      }
      if (rule.ramp !== undefined) {
        if (!Array.isArray(rule.ramp) || rule.ramp.length < 2 || rule.ramp.length > 8) err(`${where}.ramp: 2-8 colours`);
        else rule.ramp.forEach((c, j) => hex(c, `${where}.ramp[${j}]`));
      }
      if (rule.hue !== undefined && !num(rule.hue, -180, 180)) err(`${where}.hue: -180..180`);
      if (rule.saturation !== undefined && !num(rule.saturation, 0, 3)) err(`${where}.saturation: 0..3`);
      if (rule.lightness !== undefined && !num(rule.lightness, -100, 100)) err(`${where}.lightness: -100..100`);
      if (rule.mix !== undefined) {
        if (!isObj(rule.mix) || !num(rule.mix.amount, 0, 1)) err(`${where}.mix: {color, amount 0..1}`);
        else hex(rule.mix.color, `${where}.mix.color`);
      }
    });
    if (palette.colors !== undefined) {
      if (!isObj(palette.colors)) err('palette.colors: {palette name: colour}');
      else {
        for (const [n, c] of Object.entries(palette.colors)) {
          if (!names.has(n)) err(`palette.colors: "${n}" is not a palette name`);
          hex(c, `palette.colors.${n}`);
        }
      }
    }
    const rows = table.filter((r) => r.role === 'palette');
    const unmapped = rows.filter((r) => !(palette.colors && palette.colors[r.name])
      && !palette.rules.some((rule) => isObj(rule) && ruleMatches(rule, r)));
    if (unmapped.length) err(`palette: not mapped: ${unmapped.map((r) => r.name).join(', ')}`);
  }
  // the rest of the look
  if (theme.shade !== undefined) {
    if (!isObj(theme.shade)) err('shade: {factor} or {color}');
    else {
      keys(theme.shade, ['factor', 'color'], 'shade');
      if (theme.shade.factor !== undefined && !num(theme.shade.factor, 0.2, 1)) err('shade.factor: 0.2..1');
      if (theme.shade.color !== undefined) hex(theme.shade.color, 'shade.color');
    }
  }
  hex(theme.ink, 'ink');
  const clay = theme.clay;
  if (!isObj(clay)) err('clay: {wall, hatch, face?, roof?, hatch_shade?}');
  else {
    keys(clay, ['wall', 'hatch', 'face', 'roof', 'hatch_shade'], 'clay');
    hex(clay.wall, 'clay.wall');
    hex(clay.hatch, 'clay.hatch');
    for (const k of ['face', 'roof']) if (clay[k] !== undefined) hex(clay[k], `clay.${k}`);
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
  if (theme.locked_frame !== undefined) hex(theme.locked_frame, 'locked_frame');
  if (theme.gradient !== undefined && theme.gradient !== null) {
    const g = theme.gradient;
    if (!isObj(g) || !num(g.strength, 0, 1)) err('gradient: {top, bottom, strength 0..1} or null');
    else {
      hex(g.top, 'gradient.top');
      hex(g.bottom, 'gradient.bottom');
    }
  }
  if (errors.length) return { ok: false, errors, warnings, minClayDeltaE: null, closest: null };

  // honesty: clay never passes for a photographed surface
  const t = resolveTheme(theme);
  const { byName } = buildColorMap(t, table);
  const observed = table.filter((r) => ['palette', 'shade', 'ink'].includes(r.role)).map((r) => [r.name, byName.get(r.name)]);
  const face = hexToRgb(t.clay.face);
  const clays = [
    ['clay.wall', hexToRgb(t.clay.wall)],
    ['clay.hatch', hexToRgb(t.clay.hatch)],
    ['clay.face', face],
    ['clay.roof', hexToRgb(t.clay.roof)],
    ['clay.face hatch', scaleLinear(face, t.clay.hatch_shade)],
  ];
  let min = Infinity;
  let closest = null;
  for (const [cond, k] of [['如设定', [1, 1, 1]], ...lightTints(t)]) {
    for (const [cn, c] of clays) {
      for (const [on, o] of observed) {
        const d = deltaE00Rgb(tinted(c, k), tinted(o, k));
        if (d < min) {
          min = d;
          closest = [cn, on, cond];
        }
      }
    }
  }
  if (min < MIN_CLAY_DELTA_E) {
    err(`honesty: ${closest[0]} is only ΔE00 ${min.toFixed(1)} from ${closest[1]} as this theme maps it (${closest[2]}) `
      + `(clay must stay >= ${MIN_CLAY_DELTA_E} from every colour an observed facade can show)`);
  }
  const hatchWall = deltaE00Rgb(hexToRgb(t.clay.wall), hexToRgb(t.clay.hatch));
  const hatchFace = deltaE00Rgb(face, scaleLinear(face, t.clay.hatch_shade));
  if (hatchWall < MIN_HATCH_DELTA_E) err(`honesty: clay.hatch is only ΔE00 ${hatchWall.toFixed(1)} from clay.wall (hatch invisible)`);
  if (hatchFace < MIN_HATCH_DELTA_E) err(`honesty: hatch_shade leaves the face hatch only ΔE00 ${hatchFace.toFixed(1)} from clay.face`);
  return { ok: errors.length === 0, errors, warnings, minClayDeltaE: Math.round(min * 10) / 10, closest };
}

// ------------------------------------------------------------------ loading

/** CSS for the page behind the canvas: {background, clay (for the legend chip)}. */
export function themeCss(theme) {
  const bg = theme.background;
  const background = bg.type === 'solid'
    ? bg.color
    : `linear-gradient(180deg, ${bg.stops.map((s) => `${s.color} ${Math.round(s.at * 100)}%`).join(', ')})`;
  return { background, clay: theme.clay.wall };
}

/**
 * The theme a view asked for, unvalidated (the viewer validates it against the model's colour table):
 * {theme, error}. 'default' needs no request; a missing or malformed file falls back to the default
 * with an error message. fetchJson(url) -> parsed JSON (throws on failure).
 */
export async function loadTheme(id, fetchJson, base = 'themes/') {
  if (!id || id === 'default') return { theme: DEFAULT_THEME, error: null };
  if (!THEME_ID_RE.test(id)) return { theme: DEFAULT_THEME, error: `画风名称无效：${id}` };
  try {
    const theme = await fetchJson(`${base}${id}.json`);
    if (!theme || theme.id !== id) return { theme: DEFAULT_THEME, error: `画风文件 ${id}.json 的 id 不是 ${id}` };
    return { theme, error: null };
  } catch (e) {
    return { theme: DEFAULT_THEME, error: `画风 ${id} 加载失败：${e.message ?? e}` };
  }
}
