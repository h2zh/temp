// Run from the repo root: node --test 'web/tests/*.test.mjs'
// The light / dark switch (D32): web/boot.js's theme part run in a node:vm context with a fake window; the switch
// web/ui.js mountPrefs draws; and the colours themselves: every text pair of the pages at least 4.5:1 (WCAG AA) in
// both modes, the clay viewer's panels over the model, and the colours drawn into the point-cloud viewer.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { afterEach, test } from 'node:test';
import vm from 'node:vm';

import { srgbToLinear } from '../themes.js';
import { mountPrefs } from '../ui.js';
import { inLang, installDom, text } from './fake-dom.mjs';

const WEB = new URL('../', import.meta.url);
const read = (f) => readFileSync(new URL(f, WEB), 'utf8');
const SRC = read('boot.js');

// ---------- boot.js: the theme part ----------

const BLOCKED = {
  getItem() {
    throw new Error('SecurityError');
  },
  setItem() {
    throw new Error('SecurityError');
  },
  removeItem() {
    throw new Error('SecurityError');
  },
};

/**
 * Runs boot.js as a page would: the system's mode (dark), what storage holds (stored), blocked storage, and
 * a browser whose MediaQueryList only has the old addListener (oldSafari) or no matchMedia at all (noMatchMedia).
 */
function boot({ dark = false, stored, blocked = false, oldSafari = false, noMatchMedia = false } = {}) {
  const root = { lang: '', dataset: {}, classList: { add() {}, remove() {}, contains: () => false } };
  const winListeners = {};
  const events = [];
  const mqListeners = [];
  const query = { media: '(prefers-color-scheme: dark)', matches: dark };
  if (oldSafari) query.addListener = (fn) => mqListeners.push(fn);
  else query.addEventListener = (type, fn) => type === 'change' && mqListeners.push(fn);
  const store = new Map(stored === undefined ? [] : [['crowd3d.theme', stored]]);
  const queries = [];
  class CustomEvent {
    constructor(type, init) {
      this.type = type;
      this.detail = init?.detail;
    }
  }
  const w = {
    name: '',
    navigator: { languages: ['zh-CN'] },
    addEventListener: (type, fn) => (winListeners[type] ??= []).push(fn),
    dispatchEvent: (e) => events.push(e),
    CustomEvent,
    document: { documentElement: root, addEventListener() {} },
    location: { reload() {} },
    localStorage: blocked
      ? BLOCKED
      : { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) },
  };
  if (!noMatchMedia)
    w.matchMedia = (q) => {
      queries.push(q);
      return query;
    };
  vm.runInNewContext(SRC, { window: w });
  return {
    w,
    root,
    store,
    queries,
    prefs: w.crowd3dPrefs,
    /** The theme events so far, as [theme, choice]. */
    events: () => events.filter((e) => e.type === 'crowd3d:theme').map((e) => [e.detail.theme, e.detail.choice]),
    /** The system switches between light and dark while the page is open. */
    system(isDark) {
      query.matches = isDark;
      for (const fn of mqListeners) fn({ matches: isDark, media: query.media });
    },
    /** Another tab of the site wrote (or removed) the choice: the storage event this tab gets. */
    otherTab(value, key = 'crowd3d.theme') {
      if (key === 'crowd3d.theme') value === null ? store.delete(key) : store.set(key, value);
      for (const fn of winListeners.storage ?? []) fn({ key, newValue: value });
    },
    pageshow: (persisted) => (winListeners.pageshow ?? []).forEach((fn) => fn({ persisted })),
  };
}

test('by default the theme follows the system, live: <html data-theme> before the first paint, then every change', () => {
  const light = boot({ dark: false });
  assert.deepEqual(light.queries, ['(prefers-color-scheme: dark)']);
  assert.equal(light.root.dataset.theme, 'light');
  assert.equal(light.root.dataset.themeChoice, 'system');
  assert.equal(light.prefs.themeChoice(), 'system');
  assert.deepEqual(light.events(), [], 'no event for the first paint');
  light.system(true);
  assert.equal(light.root.dataset.theme, 'dark');
  light.system(false);
  assert.equal(light.root.dataset.theme, 'light');
  assert.deepEqual(light.events(), [['dark', 'system'], ['light', 'system']]);

  const dark = boot({ dark: true });
  assert.equal(dark.root.dataset.theme, 'dark');
  // a browser whose MediaQueryList only has addListener (Safari before 14) follows too
  const old = boot({ dark: true, oldSafari: true });
  assert.equal(old.root.dataset.theme, 'dark');
  old.system(false);
  assert.equal(old.root.dataset.theme, 'light');
  // no matchMedia at all: light
  assert.equal(boot({ noMatchMedia: true }).root.dataset.theme, 'light');
});

test('an explicit choice beats the system and survives its changes; a junk stored value counts as the system', () => {
  for (const [stored, dark, theme] of [
    ['light', true, 'light'],
    ['dark', false, 'dark'],
    ['system', true, 'dark'],
    ['system', false, 'light'],
  ]) {
    const b = boot({ stored, dark });
    assert.equal(b.root.dataset.theme, theme, `${stored} on a ${dark ? 'dark' : 'light'} system`);
    assert.equal(b.prefs.themeChoice(), stored);
    if (stored !== 'system') {
      b.system(!dark);
      b.system(dark);
      assert.equal(b.root.dataset.theme, theme, `${stored} stays when the system changes`);
      assert.deepEqual(b.events(), [], 'nothing changed: no event');
    }
  }
  for (const junk of ['Dark', 'auto', '', 'null', ' dark']) {
    const b = boot({ stored: junk, dark: true });
    assert.equal(b.prefs.themeChoice(), 'system', JSON.stringify(junk));
    assert.equal(b.root.dataset.theme, 'dark', JSON.stringify(junk));
  }
  const { resolveTheme } = boot().prefs;
  assert.equal(resolveTheme('dark', false), 'dark');
  assert.equal(resolveTheme('light', true), 'light');
  assert.equal(resolveTheme('system', true), 'dark');
  assert.equal(resolveTheme('system', false), 'light');
});

test('setTheme saves the choice (system removes the entry) and applies it at once; an unknown choice throws', () => {
  const b = boot({ dark: false });
  assert.equal(b.prefs.THEME_KEY, 'crowd3d.theme');
  assert.deepEqual([...b.prefs.THEME_CHOICES], ['system', 'light', 'dark']);
  assert.equal(b.prefs.setTheme('dark'), 'dark');
  assert.equal(b.store.get('crowd3d.theme'), 'dark');
  assert.equal(b.root.dataset.theme, 'dark');
  assert.equal(b.root.dataset.themeChoice, 'dark');
  b.system(true); // no change while dark is chosen
  assert.equal(b.prefs.setTheme('system'), 'dark', 'back to the system, which is dark now');
  assert.equal(b.store.has('crowd3d.theme'), false);
  b.system(false);
  assert.equal(b.root.dataset.theme, 'light', 'following the system again');
  assert.equal(b.prefs.setTheme('light'), 'light');
  assert.equal(b.store.get('crowd3d.theme'), 'light');
  for (const bad of ['Dark', 'auto', '', undefined, null]) assert.throws(() => b.prefs.setTheme(bad), /unknown theme choice/);
  assert.equal(b.store.get('crowd3d.theme'), 'light');
  assert.equal(b.root.dataset.theme, 'light');
});

test('the crowd3d:theme event: only when the theme or the choice changes, with both', () => {
  const b = boot({ dark: false });
  b.prefs.setTheme('system'); // nothing changes
  b.system(false);
  assert.deepEqual(b.events(), []);
  b.prefs.setTheme('light'); // the same theme, another choice: the switches of the page show it
  b.prefs.setTheme('light');
  b.prefs.setTheme('dark');
  b.prefs.setTheme('dark');
  b.system(true);
  b.prefs.setTheme('system'); // the system is dark now: the theme stays, the choice changes
  b.system(false);
  assert.deepEqual(b.events(), [
    ['light', 'light'],
    ['dark', 'dark'],
    ['dark', 'system'],
    ['light', 'system'],
  ]);
});

test('blocked storage: the system mode, and a choice kept for as long as the page', () => {
  const b = boot({ blocked: true, dark: true });
  assert.equal(b.root.dataset.theme, 'dark');
  assert.equal(b.prefs.themeChoice(), 'system');
  assert.equal(b.prefs.setTheme('light'), 'light');
  assert.equal(b.prefs.themeChoice(), 'light');
  b.system(false);
  b.system(true);
  assert.equal(b.root.dataset.theme, 'light');
  b.prefs.setTheme('system');
  assert.equal(b.prefs.themeChoice(), 'system');
  assert.equal(b.root.dataset.theme, 'dark');
});

test('another tab: its choice applies here at once (storage event); Back from the cache applies the stored choice', () => {
  const b = boot({ dark: false });
  b.otherTab('dark');
  assert.equal(b.root.dataset.theme, 'dark');
  assert.equal(b.prefs.themeChoice(), 'dark');
  b.otherTab('en', 'crowd3d.lang'); // another setting: nothing to do
  assert.equal(b.root.dataset.theme, 'dark');
  b.otherTab(null); // the other tab chose the system again
  assert.equal(b.root.dataset.theme, 'light');
  b.store.set('crowd3d.theme', 'dark');
  b.otherTab(undefined, null); // the whole storage was cleared or changed (key null)
  assert.equal(b.root.dataset.theme, 'dark');
  assert.deepEqual(b.events(), [['dark', 'dark'], ['light', 'system'], ['dark', 'dark']]);
  // a page restored from the back/forward cache, the choice changed on another page meanwhile
  b.store.set('crowd3d.theme', 'light');
  b.pageshow(false);
  assert.equal(b.root.dataset.theme, 'dark', 'a fresh load ran boot.js just now');
  b.pageshow(true);
  assert.equal(b.root.dataset.theme, 'light');
});

test('the language part is unchanged beside the theme part', () => {
  const b = boot({ dark: true });
  assert.deepEqual(Object.keys(b.prefs).sort(), [
    'LANG_CHOICES',
    'LANG_KEY',
    'THEME_CHOICES',
    'THEME_KEY',
    'langChoice',
    'resolveLang',
    'resolveTheme',
    'setLang',
    'setTheme',
    'themeChoice',
  ]);
  assert.equal(b.root.dataset.lang, 'zh');
  assert.equal(b.root.lang, 'zh-CN');
  b.prefs.setTheme('light');
  assert.equal(b.root.dataset.lang, 'zh');
});

// ---------- the switch (web/ui.js mountPrefs) ----------

let doc = null;
afterEach(() => {
  delete globalThis.crowd3dPrefs;
  delete globalThis.addEventListener;
  doc?.restore();
  doc = null;
});

function fakePrefs({ lang = 'system', theme = 'system' } = {}) {
  const calls = [];
  const state = { theme };
  const listeners = [];
  globalThis.addEventListener = (type, fn) => listeners.push([type, fn]);
  globalThis.crowd3dPrefs = {
    langChoice: () => lang,
    setLang: (c) => calls.push(['lang', c]),
    themeChoice: () => state.theme,
    setTheme: (c) => calls.push(['theme', c]),
  };
  return {
    calls,
    /** boot.js after a choice made elsewhere (another tab): the stored choice changes, then the event. */
    changed(choice) {
      state.theme = choice;
      for (const [type, fn] of listeners) if (type === 'crowd3d:theme') fn({ detail: { choice } });
    },
  };
}

test('the switch: a labelled language select and a labelled theme select in one group; changes call setLang / setTheme', () => {
  doc = installDom();
  const fake = fakePrefs({ lang: 'en', theme: 'dark' });
  const host = doc.getElementById('prefs');
  const group = mountPrefs(host);
  assert.equal(host.children.length, 1);
  assert.equal(group.getAttribute('role'), 'group');
  assert.equal(group.getAttribute('aria-label'), '显示设置');
  const prefs = group.querySelectorAll('.pref');
  assert.equal(prefs.length, 2);
  const [lang, theme] = prefs.map((p) => ({ label: p.querySelector('label'), select: p.querySelector('select'), icon: p.querySelector('.pref-icon') }));
  for (const p of [lang, theme]) {
    assert.equal(p.label.getAttribute('for'), p.select.getAttribute('id'));
    assert.equal(p.label.className, 'visually-hidden');
    assert.equal(p.icon.getAttribute('aria-hidden'), 'true');
    assert.match(p.icon.innerHTML, /^<svg [^>]*focusable="false"/);
    assert.equal(p.select.getAttribute('autocomplete'), 'off');
  }
  assert.deepEqual([text(lang.label), text(theme.label)], ['语言', '外观']);
  // both on 跟随系统 the two pills read the same: each select's tooltip names its setting
  assert.deepEqual([lang.select.getAttribute('title'), theme.select.getAttribute('title')], ['语言', '外观']);
  assert.equal(theme.select.getAttribute('id'), 'pref-theme');
  assert.deepEqual(theme.select.options.map((o) => o.getAttribute('value')), ['system', 'light', 'dark']);
  assert.deepEqual(theme.select.options.map((o) => o.textContent), ['跟随系统', '浅色', '深色']);
  assert.deepEqual(theme.select.options.filter((o) => o.hasAttribute('selected')).map((o) => o.getAttribute('value')), ['dark']);
  assert.equal(theme.select.value, 'dark');
  assert.equal(lang.select.value, 'en');
  theme.select.value = 'light';
  theme.select.dispatch('change');
  lang.select.value = 'zh';
  lang.select.dispatch('change');
  assert.deepEqual(fake.calls, [
    ['theme', 'light'],
    ['lang', 'zh'],
  ]);
  // a choice made in another tab: boot.js applies it and tells the page; the select shows it
  fake.changed('system');
  assert.equal(theme.select.value, 'system');
});

test('the switch in English', () => {
  doc = installDom();
  fakePrefs();
  const group = inLang('en', () => mountPrefs(doc.getElementById('prefs')));
  assert.equal(group.getAttribute('aria-label'), 'Display');
  const theme = group.querySelectorAll('.pref')[1];
  assert.equal(text(theme.querySelector('label')), 'Appearance');
  assert.equal(theme.querySelector('select').getAttribute('title'), 'Appearance');
  assert.equal(group.querySelectorAll('.pref')[0].querySelector('select').getAttribute('title'), 'Language');
  assert.deepEqual(theme.querySelector('select').options.map((o) => o.textContent), ['System', 'Light', 'Dark']);
});

test('without boot.js (window.crowd3dPrefs) the switch renders nothing', () => {
  doc = installDom();
  const host = doc.getElementById('prefs');
  assert.equal(mountPrefs(host), null);
  assert.equal(host.children.length, 0);
});

// ---------- the CSS ----------

const STYLE = read('style.css');
const STYLIZED = read('stylized.html');
const PHOTOPOSES = read('photoposes.js');
const COMPARE = read('compare.js');
const noComments = (css) => css.replace(/\/\*[\s\S]*?\*\//g, '');

/** The declarations of the first rule whose selector is exactly `selector`, as {name: value}. */
function ruleBlock(css, selector) {
  const src = noComments(css);
  const re = new RegExp(`(^|[}\\s])${selector.replace(/[[\]().*+?^$|\\]/g, '\\$&')}\\s*\\{([^}]*)\\}`);
  const m = src.match(re);
  assert.ok(m, `a rule ${selector}`);
  return Object.fromEntries([...m[2].matchAll(/(--[\w-]+|[\w-]+)\s*:\s*([^;]+);?/g)].map((d) => [d[1], d[2].trim()]));
}

/** Every rule of a stylesheet (those inside @media too), in order: {sels: [selector, ...], decls: {name: value}}. */
function rulesOf(css) {
  return [...noComments(css).matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({
    sels: m[1].split(',').map((x) => x.trim().replace(/\s+/g, ' ')),
    decls: Object.fromEntries([...m[2].matchAll(/(--[\w-]+|[\w-]+)\s*:\s*([^;]+);?/g)].map((d) => [d[1], d[2].trim()])),
  }));
}
/** What the rules naming `selector` (alone or in a group) declare, later rules winning. */
function declsFor(css, selector) {
  const out = {};
  for (const r of rulesOf(css)) if (r.sels.includes(selector)) Object.assign(out, r.decls);
  return out;
}
// the properties that take a colour
const COLOUR_PROP = /^(color|background(-color)?|border(-(top|right|bottom|left))?(-color)?|outline(-color)?|box-shadow|text-shadow|fill|stroke|caret-color|accent-color|text-decoration(-color)?|column-rule(-color)?)$/;
// a colour written out: #hex, rgb()/hsl(), or a CSS colour name
const LITERAL_COLOUR =
  /#[0-9a-f]{3,8}(?![\w-])|\b(rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(|(?<![\w-])(white|black|gr[ae]y|silver|red|maroon|green|lime|blue|navy|yellow|orange|purple|pink|brown|cyan|aqua|teal|olive|fuchsia|magenta|gold|beige|ivory|khaki|lavender|tan|salmon|coral|crimson|indigo|violet)(?![\w-])/i;
/** The colour declarations outside the :root token blocks that write a colour out: ['selector { prop: value }', ...]. */
function literalColours(css) {
  const out = [];
  for (const r of rulesOf(css)) {
    if (r.sels.every((x) => /^:root(\[data-theme="dark"\])?$/.test(x))) continue;
    for (const [k, v] of Object.entries(r.decls))
      if (COLOUR_PROP.test(k) && LITERAL_COLOUR.test(v)) out.push(`${r.sels.join(', ')} { ${k}: ${v} }`);
  }
  return out;
}

const STYLIZED_CSS = STYLIZED.slice(STYLIZED.indexOf('<style>'), STYLIZED.indexOf('</style>'));
const MODES = {
  light: ruleBlock(STYLE, ':root'),
  dark: { ...ruleBlock(STYLE, ':root'), ...ruleBlock(STYLE, ':root[data-theme="dark"]') },
};
const PANEL = {
  light: ruleBlock(STYLIZED_CSS, ':root'),
  dark: { ...ruleBlock(STYLIZED_CSS, ':root'), ...ruleBlock(STYLIZED_CSS, ':root[data-theme="dark"]') },
};

/** '#rgb' | '#rrggbb' | 'rgb(a)(r, g, b[, a])' -> [r, g, b, a] (0..255, alpha 0..1). */
function parseColor(value) {
  const v = String(value).trim();
  let m = v.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (m) {
    const h = m[1].length === 3 ? [...m[1]].map((c) => c + c).join('') : m[1];
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)).concat(1);
  }
  m = v.match(/^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)$/i);
  assert.ok(m, `a colour: ${v}`);
  return [Number(m[1]), Number(m[2]), Number(m[3]), m[4] === undefined ? 1 : Number(m[4])];
}

/** A colour with alpha painted over an opaque one. */
const over = (top, under) => {
  const [r, g, b, a] = parseColor(top);
  const u = parseColor(under);
  return `rgb(${[r, g, b].map((c, i) => Math.round(c * a + u[i] * (1 - a))).join(', ')})`;
};
const luminance = (value) => {
  const [r, g, b, a] = parseColor(value);
  assert.equal(a, 1, `an opaque colour: ${value}`);
  const [R, G, B] = [r, g, b].map((c) => srgbToLinear(c / 255));
  return 0.2126 * R + 0.7152 * G + 0.0722 * B;
};
/** The WCAG 2 contrast ratio of two opaque colours. */
function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}
const token = (mode, name) => {
  const v = mode[`--${name}`];
  assert.ok(v, `--${name}`);
  return v;
};
const isColour = (v) => /^(#[0-9a-f]{3,8}|rgba?\()/i.test(v);
const colourKeys = (block) => Object.keys(block).filter((k) => k.startsWith('--') && isColour(block[k])).sort();

test('the contrast helper: WCAG 2 reference values', () => {
  assert.equal(contrast('#000000', '#ffffff').toFixed(2), '21.00');
  assert.equal(contrast('#fff', '#fff'), 1);
  assert.equal(contrast('#767676', '#ffffff').toFixed(2), '4.54'); // the darkest grey that passes on white
  assert.equal(contrast('#777777', '#ffffff').toFixed(2), '4.48');
  assert.equal(over('rgba(0, 0, 0, 0.5)', '#ffffff'), 'rgb(128, 128, 128)');
});

test('no page decides the mode by prefers-color-scheme: every dark rule is keyed on <html data-theme> (boot.js)', () => {
  for (const [f, src] of [
    ['style.css', STYLE],
    ['stylized.html', STYLIZED],
    ['photoposes.js', PHOTOPOSES],
    ['compare.js', COMPARE],
  ])
    assert.doesNotMatch(src, /prefers-color-scheme/, f);
  assert.match(STYLE, /:root\[data-theme="dark"\] \.leaflet-tile-pane \{/);
  assert.equal(ruleBlock(STYLE, ':root')['color-scheme'], 'light');
  assert.equal(ruleBlock(STYLE, ':root[data-theme="dark"]')['color-scheme'], 'dark');
  assert.equal(ruleBlock(STYLIZED_CSS, ':root')['color-scheme'], 'light');
  assert.equal(ruleBlock(STYLIZED_CSS, ':root[data-theme="dark"]')['color-scheme'], 'dark');
});

test('the stylesheets write no colour out: outside the :root token blocks every colour is a token', () => {
  // the helper itself: what it flags and what it lets through
  assert.deepEqual(literalColours('.a { color: #fff; background: rgba(0, 0, 0, .5); border: 1px solid white; fill: Red; }').length, 4);
  assert.deepEqual(
    literalColours(':root { --x: #fff; } .a { white-space: nowrap; color: var(--x); border: 2px solid currentColor; background: transparent; }'),
    [],
  );
  assert.deepEqual(literalColours(STYLE), [], 'style.css');
  assert.deepEqual(literalColours(STYLIZED_CSS), [], 'stylized.html');
  // stylized.html: the art theme's tokens (the sky, --ink, --green) only on the page itself (body: the sky and the
  // marks over the model) and on the model pane while comparing (its sky, D32); every other colour is a panel token,
  // so every panel follows light / dark
  const offPanel = [];
  for (const r of rulesOf(STYLIZED_CSS)) {
    if (r.sels.every((x) => /^:root(\[data-theme="dark"\])?$/.test(x)) || r.sels.join() === 'body') continue;
    if (r.sels.join() === 'body.pp-compare #stage' && Object.keys(r.decls).join() === 'background' && r.decls.background === 'var(--sky)') continue;
    for (const [k, v] of Object.entries(r.decls))
      for (const m of v.matchAll(/var\((--[\w-]+)/g)) if (COLOUR_PROP.test(k) && !m[1].startsWith('--panel')) offPanel.push(`${r.sels} { ${k}: ${v} }`);
  }
  assert.deepEqual(offPanel, []);
  assert.deepEqual(declsFor(STYLIZED_CSS, 'body').color, 'var(--ink)');
  // the sky is the art theme's in both places (stylized.js sets --sky to the theme's background)
  assert.equal(declsFor(STYLIZED_CSS, 'body').background, 'var(--sky)');
  assert.equal(declsFor(STYLIZED_CSS, 'body.pp-compare #stage').background, 'var(--sky)');
  assert.match(ruleBlock(STYLIZED_CSS, ':root')['--sky'], /^linear-gradient\(180deg, var\(--sky-top\) 0%, var\(--sky-mid\) 55%, var\(--sky-low\) 100%\)$/);
  assert.ok(!('--sky' in ruleBlock(STYLIZED_CSS, ':root[data-theme="dark"]')), 'the sky does not follow light / dark');
  assert.match(read('stylized.js'), /document\.documentElement\.style\.setProperty\('--sky', themeCss\(look\)\.background\)/);
});

test('light and dark define the same colour tokens (style.css; stylized.html panels)', () => {
  const light = colourKeys(ruleBlock(STYLE, ':root'));
  assert.ok(light.length >= 20, light.join(' '));
  assert.deepEqual(colourKeys(ruleBlock(STYLE, ':root[data-theme="dark"]')), light);
  const panelLight = colourKeys(ruleBlock(STYLIZED_CSS, ':root')).filter((k) => k.startsWith('--panel'));
  assert.deepEqual(panelLight, ['--panel', '--panel-chip', '--panel-fg', '--panel-go', '--panel-line', '--panel-note', '--panel-page', '--panel-soft']);
  assert.deepEqual(colourKeys(ruleBlock(STYLIZED_CSS, ':root[data-theme="dark"]')), panelLight);
});

// Every text colour of style.css on the background it is drawn on: [fg token, bg token, where]. Measured in both
// modes, at least 4.5:1 (WCAG 2 AA for body text; the pages have no text that would only need 3:1).
const PAIRS = [
  ['text', 'bg', 'page text'],
  ['text', 'surface', 'cards, popups, inputs'],
  ['text', 'surface-2', 'contributor chips, hovered zoom buttons'],
  ['text', 'accent-soft', 'the chosen model in the list'],
  ['text', 'warn-soft', 'the agreement draft notice'],
  ['muted', 'bg', 'intro lines on the page'],
  ['muted', 'surface', 'hints and meta lines, map attribution'],
  ['muted', 'surface-2', 'muted badge, the settings icons'],
  ['muted', 'accent-soft', 'meta line of the chosen model'],
  ['muted', 'viewer-bg', 'viewer messages (the edge of the gradient)'],
  ['link', 'bg', 'links on the page'],
  ['link', 'surface', 'links in cards and popups'],
  ['link', 'surface-2', 'links on grey'],
  ['link', 'accent-soft', 'links in the chosen model'],
  ['accent', 'surface', 'the stylized-model link'],
  ['accent', 'info-soft', 'running job chip'],
  ['accent-text', 'accent', 'primary buttons'],
  ['ok', 'ok-soft', 'done chips and badges'],
  ['ok', 'surface', 'form status: done'],
  ['warn', 'warn-soft', 'queued chips, partial badges'],
  ['bad', 'bad-soft', 'error notices, failed chips'],
  ['bad', 'surface', 'form errors, job errors, compass N'],
  ['bad', 'surface-2', 'north label on the compass core'],
  ['on-partial', 'sector-partial', 'compass count on a partial side'],
  ['on-face', 'sector-face', 'compass count and check on a finished side'],
  ['surface', 'bad', 'the ! of a failed side'],
];
// Colours of style.css that are not text: wedge, swatch and core fills, and the ★ before a full model's name
const NON_TEXT = new Set(['sector-locked', 'sector-partial', 'sector-face', 'surface-2']);

test('every text colour pair is at least 4.5:1 in light and in dark (WCAG AA)', () => {
  const report = [];
  for (const [name, mode] of Object.entries(MODES))
    for (const [fg, bg, where] of PAIRS) {
      const r = contrast(token(mode, fg), token(mode, bg));
      if (r < 4.5) report.push(`${name}: --${fg} on --${bg} (${where}) ${r.toFixed(2)}:1`);
    }
  assert.deepEqual(report, []);
});

test('every text colour of style.css is in PAIRS: color and SVG fill tokens are measured or not text', () => {
  const fgs = new Set(PAIRS.map(([fg]) => fg));
  const css = noComments(STYLE);
  const used = new Set([...css.matchAll(/(?<![-\w])(?:color|fill):\s*var\(--([\w-]+)\)/g)].map((m) => m[1]));
  assert.ok(used.size >= 10, [...used].join(' '));
  const unmeasured = [...used].filter((t) => !fgs.has(t) && !NON_TEXT.has(t));
  assert.deepEqual(unmeasured, [], 'a new text colour: add its pairs to PAIRS');
  for (const t of NON_TEXT) assert.ok(!fgs.has(t), `${t} is text after all`);
  // the graphics a reader needs: a finished side's wedge and the focus ring at least 3:1 (WCAG 1.4.11)
  for (const mode of Object.values(MODES))
    for (const [fg, bg] of [
      ['sector-face', 'surface'],
      ['sector-face', 'bg'],
      ['focus', 'surface'],
      ['focus', 'bg'],
    ])
      assert.ok(contrast(token(mode, fg), token(mode, bg)) >= 3, `--${fg} on --${bg}`);
});

// What leaflet.css (1.9.4) paints white or grey on the home page's map, and the tokens style.css paints it with:
// [selector, {property: value}, [text token, the token it is drawn on] or null]. The text pair is one of PAIRS.
const LEAFLET = [
  ['.leaflet-popup-content-wrapper', { background: 'var(--surface)', color: 'var(--text)' }, ['text', 'surface']],
  ['.leaflet-popup-tip', { background: 'var(--surface)' }, null],
  ['.leaflet-tooltip', { background: 'var(--surface)', color: 'var(--text)', 'border-color': 'var(--border)' }, ['text', 'surface']],
  ['.leaflet-tooltip-top::before', { 'border-top-color': 'var(--surface)' }, null],
  ['.leaflet-tooltip-bottom::before', { 'border-bottom-color': 'var(--surface)' }, null],
  ['.leaflet-tooltip-left::before', { 'border-left-color': 'var(--surface)' }, null],
  ['.leaflet-tooltip-right::before', { 'border-right-color': 'var(--surface)' }, null],
  ['.leaflet-container .leaflet-control-attribution', { background: 'var(--surface)', color: 'var(--muted)' }, ['muted', 'surface']],
  ['.leaflet-container .leaflet-control-attribution a', { color: 'var(--link)' }, ['link', 'surface']],
  ['.leaflet-container .map-popup a', { color: 'var(--link)' }, ['link', 'surface']],
  ['.leaflet-container a.leaflet-popup-close-button', { color: 'var(--muted)' }, ['muted', 'surface']],
  ['.leaflet-container a.leaflet-popup-close-button:hover', { color: 'var(--text)' }, ['text', 'surface']],
  ['.leaflet-container a.leaflet-popup-close-button:focus', { color: 'var(--text)' }, ['text', 'surface']],
  ['.leaflet-bar a', { 'background-color': 'var(--surface)', 'border-bottom-color': 'var(--border)', color: 'var(--text)' }, ['text', 'surface']],
  ['.leaflet-bar a:hover', { 'background-color': 'var(--surface-2)' }, ['text', 'surface-2']],
  ['.leaflet-bar a:focus', { 'background-color': 'var(--surface-2)' }, ['text', 'surface-2']],
  ['.leaflet-bar a.leaflet-disabled', { 'background-color': 'var(--surface-2)', color: 'var(--muted)' }, ['muted', 'surface-2']],
  [':root[data-theme="dark"] .leaflet-container', { background: 'var(--surface-2)' }, null],
];

test("Leaflet's white surfaces, the placeholder and the agreement's marks are painted with tokens (measured in PAIRS)", () => {
  const measured = new Set(PAIRS.map(([fg, bg]) => `${fg}/${bg}`));
  const rows = [
    ...LEAFLET,
    // the browser's placeholder grey is 3.75:1 on the dark surface; inputs are --surface
    ['::placeholder', { color: 'var(--muted)', opacity: '1' }, ['muted', 'surface']],
    // the agreement's open points: the browser's <mark> is a bright yellow block in dark mode
    ['.terms mark', { background: 'var(--warn-soft)', color: 'var(--text)' }, ['text', 'warn-soft']],
  ];
  for (const [sel, want, pair] of rows) {
    const got = declsFor(STYLE, sel);
    for (const [k, v] of Object.entries(want)) assert.equal(got[k], v, `${sel} ${k}`);
    if (pair) assert.ok(measured.has(pair.join('/')), `${sel}: --${pair[0]} on --${pair[1]} is in PAIRS`);
  }
});

test('the coverage compass: a hovered wedge (lighter over the card) keeps its count at least 4.5:1 in both modes', () => {
  const hover = ruleBlock(STYLE, '.compass-sector:hover .compass-wedge');
  assert.deepEqual(Object.keys(hover), ['opacity'], 'another hover effect: measure it here');
  const alpha = Number(hover.opacity);
  assert.ok(alpha > 0 && alpha <= 1, hover.opacity);
  const withAlpha = (hex, a) => `rgba(${parseColor(hex).slice(0, 3).join(', ')}, ${a})`;
  const report = [];
  for (const [name, mode] of Object.entries(MODES))
    for (const [fg, wedge] of [
      ['on-face', 'sector-face'],
      ['on-partial', 'sector-partial'],
    ])
      for (const under of ['surface', 'bg']) {
        const r = contrast(token(mode, fg), over(withAlpha(token(mode, wedge), alpha), token(mode, under)));
        if (r < 4.5) report.push(`${name}: --${fg} on hovered --${wedge} over --${under} ${r.toFixed(2)}:1`);
      }
  assert.deepEqual(report, []);
});

test('the clay viewer: its panels are readable over any model, in both modes; the in-scene marks stay the art theme', () => {
  for (const [name, mode] of Object.entries(PANEL)) {
    const panel = token(mode, 'panel');
    // the panel is translucent: measured over the brightest and the darkest model behind it
    for (const under of ['#ffffff', '#000000']) {
      const bg = over(panel, under);
      for (const fg of ['panel-fg', 'panel-soft', 'panel-note']) {
        const r = contrast(token(mode, fg), bg);
        assert.ok(r >= 4.5, `${name}: --${fg} on --panel over ${under}: ${r.toFixed(2)}`);
      }
    }
    for (const bg of ['panel-chip', 'panel-go', 'panel-page']) {
      const r = contrast(token(mode, 'panel-fg'), token(mode, bg));
      assert.ok(r >= 4.5, `${name}: --panel-fg on --${bg}: ${r.toFixed(2)}`);
    }
  }
  // the marks drawn over the model belong to the art theme (画风): one ink in both modes
  assert.equal(token(PANEL.light, 'ink'), '#2d2a32');
  assert.ok(!('--ink' in ruleBlock(STYLIZED_CSS, ':root[data-theme="dark"]')), 'the dark block leaves --ink alone');
  // the ring distance labels: #4b5a44 on white .85 over anything
  const ring = PHOTOPOSES.match(/\.pp-ring\{[^}]*color:(#[0-9a-f]{6});background:(rgba\([^)]*\))/i);
  assert.ok(ring, '.pp-ring');
  for (const under of ['#ffffff', '#000000']) assert.ok(contrast(ring[1], over(ring[2], under)) >= 4.5, `.pp-ring over ${under}`);
  // the art themes never look at the light/dark choice
  for (const f of ['themes.js', 'stylized.js']) assert.doesNotMatch(read(f), /data-theme|dataset\.theme|crowd3d:theme|crowd3d\.theme/, f);
});

test('the clay viewer panels use the panel tokens: stylized.html and photoposes.js', () => {
  // stylized.html: the panels, the face card and the status take their colours from --panel-*
  for (const sel of ['#panel', '#pick', '#status']) {
    const rule = ruleBlock(STYLIZED_CSS, sel);
    assert.equal(rule.background, 'var(--panel)', sel);
    assert.equal(rule.color, 'var(--panel-fg)', sel);
    assert.match(rule.border, /var\(--panel-line\)/, sel);
  }
  assert.doesNotMatch(STYLIZED_CSS, /--card\b/);
  // what sits on the panels: chips, the 画风 select, the settings, the attribution lines and the theme note
  const PANEL_RULES = [
    ['.chips li', { background: 'var(--panel-chip)', border: /var\(--panel-line\)/ }],
    ['#panel .theme-row select', { background: 'var(--panel-chip)', color: 'var(--panel-fg)', border: /var\(--panel-line\)/ }],
    ['.pref select', { background: 'var(--panel-chip)', color: 'var(--panel-fg)', border: /var\(--panel-line\)/ }],
    ['.pref-icon', { color: 'var(--panel-fg)' }],
    ['#attribution p', { color: 'var(--panel-soft)' }],
    ['#theme-note', { color: 'var(--panel-note)' }],
  ];
  for (const [sel, want] of PANEL_RULES) {
    const got = declsFor(STYLIZED_CSS, sel);
    for (const [k, v] of Object.entries(want)) {
      if (v instanceof RegExp) assert.match(got[k] ?? '', v, `${sel} ${k}`);
      else assert.equal(got[k], v, `${sel} ${k}`);
    }
  }
  // photoposes.js: every panel rule reads --panel-* (its light value only as the fallback), the marks keep --ink
  const start = PHOTOPOSES.indexOf('const CSS = `') + 'const CSS = `'.length;
  const css = PHOTOPOSES.slice(start, PHOTOPOSES.indexOf('`;', start));
  const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ sel: m[1].trim(), body: m[2] }));
  const bodyOf = (sel) => rules.filter((r) => r.sel.split(',').includes(sel)).map((r) => r.body).join(';');
  const withoutFallbacks = (body) => body.replace(/var\((--[\w-]+),[^()]*(\([^()]*\))?[^()]*\)/g, 'var($1)');
  for (const sel of ['.pp-panel', '.pp-sw', '.pp-err', '.pp-note', '.pp-tip', '.pp-card', '.pp-card .s', '.pp-card button', '.pp-card button.go']) {
    const body = bodyOf(sel);
    assert.ok(body, sel);
    assert.match(body, /var\(--panel/, sel);
    assert.doesNotMatch(withoutFallbacks(body), /#fff\b|#ffffff|#5b5763|#ffe8cc|rgba\(255,255,255|var\(--ink|var\(--card/i, sel);
  }
  for (const sel of ['.pp-panel', '.pp-card', '.pp-tip', '.pp-card button']) assert.match(bodyOf(sel), /(^|[;\s])color:var\(--panel-fg/, sel);
  for (const sel of ['.pp-bubble b', '.pp-north']) assert.match(bodyOf(sel), /var\(--ink/, sel);
  assert.doesNotMatch(css, /var\(--card/);
});

test('the photo comparison (compare.js) uses only the panel tokens, the photo-only scrim and the page backdrop (D32)', () => {
  const start = COMPARE.indexOf('const CSS = `') + 'const CSS = `'.length;
  const css = COMPARE.slice(start, COMPARE.indexOf('`;', start));
  assert.ok(css.length > 500);
  const rules = rulesOf(css);
  // every colour is a --panel-* token, without a fallback; the only colour written out is the photo-only scrim
  const off = [];
  for (const r of rules)
    for (const [k, v] of Object.entries(r.decls)) {
      if (!COLOUR_PROP.test(k) || /^(transparent|none|0)$/.test(v)) continue;
      const scrim = r.sels.join() === '.pp-cmp[data-mode="photo"]' && k === 'background' && v === 'rgba(0,0,0,.55)';
      const vars = [...v.matchAll(/var\((--[\w-]+)([^)]*)\)/g)];
      const tokensOnly = vars.length > 0 && vars.every((m) => m[1].startsWith('--panel') && m[2] === '');
      if (!scrim && (!tokensOnly || LITERAL_COLOUR.test(v.replace(/var\([^)]*\)/g, '')))) off.push(`${r.sels.join()} { ${k}: ${v} }`);
    }
  assert.deepEqual(off, []);
  assert.doesNotMatch(css, /#fff\b|#ffffff|#5b5763|#ffe8cc|var\(--ink|var\(--card|var\(--sky/i);
  // the page behind the panes, the bars, the chips, the state line and the pressed mode button
  assert.equal(declsFor(css, 'body.pp-compare').background, 'var(--panel-page)');
  for (const sel of ['.pp-cmp-top', '.pp-cmp-bar']) {
    const d = declsFor(css, sel);
    assert.equal(d.background, 'var(--panel)', sel);
    assert.equal(d.color, 'var(--panel-fg)', sel);
    assert.match(d.border, /var\(--panel-line\)/, sel);
  }
  for (const sel of ['.pp-cmp-bar button', '.pp-cmp-chip']) {
    assert.equal(declsFor(css, sel).background, 'var(--panel-chip)', sel);
    assert.equal(declsFor(css, sel).color, 'var(--panel-fg)', sel);
  }
  assert.equal(declsFor(css, '.pp-cmp-state').background, 'var(--panel)');
  assert.equal(declsFor(css, '.pp-cmp-state').color, 'var(--panel-fg)');
  assert.equal(declsFor(css, '.pp-cmp-warn').color, 'var(--panel-note)');
  assert.equal(declsFor(css, '.pp-cmp-bar button[aria-pressed="true"]').background, 'var(--panel-go)');
  assert.equal(declsFor(css, '.pp-cmp-op input')['accent-color'], 'var(--panel-fg)');
  // these pairs are the ones measured above in both modes (--panel-fg on --panel over anything, on --panel-chip,
  // --panel-go and --panel-page; --panel-note on --panel); the bars stay readable over the photo-only scrim too
  for (const [name, mode] of Object.entries(PANEL)) {
    const bg = over(token(mode, 'panel'), over('rgba(0,0,0,.55)', '#ffffff'));
    for (const fg of ['panel-fg', 'panel-note']) assert.ok(contrast(token(mode, fg), bg) >= 4.5, `${name}: --${fg} on a bar over the scrim`);
  }
  // touch screens: hit areas of at least 44 px
  assert.match(css, /@media \(pointer:coarse\)\{\.pp-cmp-bar button\{min-height:44px;min-width:44px\}\.pp-cmp-op input\{height:44px\}\}/);
});

test('the point-cloud viewer: what it draws into the scene is at least 3:1 on the light and the dark background', () => {
  const src = read('viewer.js');
  const constant = (name) => {
    const m = src.match(new RegExp(`const ${name} = (?:0x([0-9a-f]{6})|'#([0-9a-f]{6})');`, 'i'));
    assert.ok(m, name);
    return `#${m[1] ?? m[2]}`;
  };
  const colours = { MARK_COLOR: constant('MARK_COLOR'), POINT_COLOR: constant('POINT_COLOR'), NORTH_COLOR: constant('NORTH_COLOR'), NORTH_DISC: constant('NORTH_DISC') };
  assert.deepEqual(colours, { MARK_COLOR: '#ec5711', POINT_COLOR: '#768aa5', NORTH_COLOR: '#d9473f', NORTH_DISC: '#c2382e' });
  for (const [name, mode] of Object.entries(MODES))
    for (const [what, colour] of Object.entries(colours))
      for (const bg of ['surface', 'viewer-bg']) {
        const r = contrast(colour, token(mode, bg));
        assert.ok(r >= 3, `${name}: ${what} ${colour} on --${bg} ${r.toFixed(2)}`);
      }
  assert.ok(contrast('#ffffff', colours.NORTH_DISC) >= 4.5, 'the white N on its disc');
  assert.match(src, /color: geometry\.hasAttribute\('color'\) \? 0xffffff : POINT_COLOR/);
  assert.match(src, /const color = MARK_COLOR;/);
  assert.match(src, /textSprite\('N', '#ffffff', NORTH_DISC\)/);
});
