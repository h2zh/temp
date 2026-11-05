// Run from the repo root: node --test 'web/tests/*.test.mjs'
// The page language (D32): web/i18n.js and the texts in web/strings.js; that every key the pages use exists and
// every entry is used; how much Chinese is still written outside strings.js (a ratchet that only goes down); that
// every page runs web/boot.js first and has the language switch; the two language versions of the user agreement.
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { afterEach, test } from 'node:test';

import { applyDom, fmtDateTime, fmtList, fmtNum, keyed, keyedTables, lang, locale, missing, t, useLang } from '../i18n.js';
import { LANG_NAMES, STRINGS } from '../strings.js';
import { FakeElement, inLang, installDom } from './fake-dom.mjs';

const WEB = new URL('../', import.meta.url);
const read = (name) => readFileSync(new URL(name, WEB), 'utf8');
const JS_FILES = readdirSync(WEB).filter((f) => f.endsWith('.js')).sort();
const HTML_FILES = readdirSync(WEB).filter((f) => f.endsWith('.html')).sort();

// Chinese characters and CJK / full-width punctuation: none of it belongs in an English text.
const CJK = /[\p{Script=Han}　-〿＀-￯]/u;

afterEach(() => useLang(null));

// ---------- scanners ----------

/**
 * Reads a JS source: its string and template literals, comments skipped (a template's text without its ${} parts;
 * literals inside ${} count on their own; regex literals are no strings), and the source with its comments blanked.
 */
function scanJs(src) {
  const out = [];
  const comments = [];
  const n = src.length;
  let i = 0;
  let prev = ''; // the last token: '' (start), 'w' (a word), 'x' (a literal) or a punctuation character
  let word = '';
  const regexAllowed = () =>
    prev === '' ||
    (prev === 'w' ? /^(return|typeof|case|do|else|in|of|void|yield|await|delete|instanceof|new|throw)$/.test(word) : prev !== 'x' && !')]}'.includes(prev));
  function string(q) {
    let j = i + 1;
    let s = '';
    while (j < n && src[j] !== q && src[j] !== '\n') {
      if (src[j] === '\\') {
        s += src.slice(j, j + 2);
        j += 2;
        continue;
      }
      s += src[j++];
    }
    out.push(s);
    i = j + 1;
  }
  function template() {
    let j = i + 1;
    let s = '';
    while (j < n && src[j] !== '`') {
      if (src[j] === '\\') {
        s += src.slice(j, j + 2);
        j += 2;
      } else if (src[j] === '$' && src[j + 1] === '{') {
        i = j + 2;
        code(true);
        j = i;
      } else s += src[j++];
    }
    out.push(s);
    i = j + 1;
  }
  function regex() {
    let j = i + 1;
    let inClass = false;
    while (j < n && src[j] !== '\n') {
      if (src[j] === '\\') j++;
      else if (src[j] === '[') inClass = true;
      else if (src[j] === ']') inClass = false;
      else if (src[j] === '/' && !inClass) break;
      j++;
    }
    j++;
    while (j < n && /[a-z]/i.test(src[j])) j++;
    i = j;
  }
  function code(inTemplate) {
    let depth = 0;
    while (i < n) {
      const c = src[i];
      if (c === '/' && src[i + 1] === '/') {
        const j = src.indexOf('\n', i);
        comments.push([i, j < 0 ? n : j]);
        i = j < 0 ? n : j;
      } else if (c === '/' && src[i + 1] === '*') {
        const j = src.indexOf('*/', i + 2);
        comments.push([i, j < 0 ? n : j + 2]);
        i = j < 0 ? n : j + 2;
      } else if (c === "'" || c === '"') {
        string(c);
        prev = 'x';
      } else if (c === '`') {
        template();
        prev = 'x';
      } else if (c === '/' && regexAllowed()) {
        regex();
        prev = 'x';
      } else if (/[\w$]/.test(c)) {
        let j = i;
        while (j < n && /[\w$]/.test(src[j])) j++;
        word = src.slice(i, j);
        prev = 'w';
        i = j;
      } else {
        if (inTemplate && c === '{') depth++;
        if (inTemplate && c === '}' && depth-- === 0) {
          i++;
          return;
        }
        if (!/\s/.test(c)) prev = c;
        i++;
      }
    }
  }
  code(false);
  let stripped = src;
  for (const [a, b] of comments.reverse()) stripped = stripped.slice(0, a) + ' '.repeat(b - a) + stripped.slice(b);
  return { literals: out, code: stripped };
}

/** The string and template literals of a JS source (see scanJs). */
export const jsLiterals = (src) => scanJs(src).literals;

const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr']);
const decode = (s) =>
  s.replace(/&nbsp;/g, ' ').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&');

/** A small HTML tree: {tag, attrs, children, raw (script / style text)} and {text} nodes; comments dropped. */
export function parseHtml(src) {
  src = src.replace(/<!--[\s\S]*?-->/g, '');
  const root = { tag: '#root', attrs: {}, children: [], parent: null };
  let cur = root;
  let i = 0;
  const tagRe = /<(\/)?([a-zA-Z][\w-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>|<![^>]*>/g;
  for (let m; (m = tagRe.exec(src)); ) {
    if (m.index > i) cur.children.push({ text: decode(src.slice(i, m.index)) });
    i = tagRe.lastIndex;
    if (!m[2]) continue; // <!doctype>
    const tag = m[2].toLowerCase();
    if (m[1]) {
      let e = cur;
      while (e !== root && e.tag !== tag) e = e.parent;
      if (e !== root) cur = e.parent;
      continue;
    }
    const attrs = {};
    for (const a of m[3].matchAll(/([^\s=/]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) {
      attrs[a[1].toLowerCase()] = decode(a[2] ?? a[3] ?? a[4] ?? '');
    }
    const node = { tag, attrs, children: [], parent: cur };
    cur.children.push(node);
    if (tag === 'script' || tag === 'style') {
      const end = src.indexOf(`</${tag}`, i);
      node.raw = src.slice(i, end);
      i = src.indexOf('>', end) + 1;
      tagRe.lastIndex = i;
    } else if (!VOID.has(tag) && !m[3].trim().endsWith('/')) cur = node;
  }
  if (i < src.length) cur.children.push({ text: decode(src.slice(i)) });
  return root;
}

/** Every element of a parsed tree in document order. */
function* elements(node) {
  for (const c of node.children ?? []) {
    if (!c.tag) continue;
    yield c;
    yield* elements(c);
  }
}

const textOf = (node) => (node.text ?? (node.children ?? []).map(textOf).join(''));
const squash = (s) => s.replace(/\s+/g, ' ').trim();

/**
 * Chinese still written outside strings.js in a page: text runs and attribute values with CJK characters, except
 * inside [data-i18n] elements (whose text must equal the key's zh value), keyed attributes ([data-i18n-<name>], whose
 * value must equal the key's zh value) and [data-lang-only="zh"] (the Chinese version of the agreement).
 */
function htmlCjk(root, file) {
  let count = 0;
  const visit = (node, inside) => {
    for (const c of node.children) {
      if (!c.tag) {
        if (!inside && CJK.test(c.text)) count++;
        continue;
      }
      if (c.tag === 'script' || c.tag === 'style') continue;
      const zhOnly = c.attrs['data-lang-only'] === 'zh';
      for (const [name, value] of Object.entries(c.attrs)) {
        if (name.startsWith('data-i18n-')) {
          const target = name.slice('data-i18n-'.length);
          assert.equal(c.attrs[target], STRINGS[value]?.zh, `${file}: ${target} of [${name}="${value}"] is the key's zh text`);
        } else if (!name.startsWith('data-') && CJK.test(value) && !inside && !zhOnly && !(`data-i18n-${name}` in c.attrs)) count++;
      }
      const key = c.attrs['data-i18n'];
      if (key !== undefined) {
        assert.ok(!c.children.some((k) => k.tag), `${file}: [data-i18n="${key}"] holds text only`);
        assert.equal(squash(textOf(c)), STRINGS[key]?.zh, `${file}: the text of [data-i18n="${key}"] is the key's zh text`);
      }
      visit(c, inside || key !== undefined || zhOnly);
    }
  };
  visit(root, false);
  return count;
}

// ---------- the texts ----------

const placeholders = (v) =>
  [...new Set((typeof v === 'object' ? Object.values(v).join(' ') : v).match(/\{\w+\}/g) ?? [])].sort().join(',');
const EMPTY_ALLOWED = new Set(); // keys whose text may be empty in a language

test('every text has exactly zh and en with the same placeholders; English plurals have one and other; no CJK in English', () => {
  assert.ok(Object.keys(STRINGS).length > 0);
  for (const [key, entry] of Object.entries(STRINGS)) {
    assert.match(key, /^[a-z]\w*(\.[\w-]+)+$/i, key); // dotted; a keyed() code may hold '-'
    assert.deepEqual(Object.keys(entry).sort(), ['en', 'zh'], key);
    for (const l of ['zh', 'en']) {
      const v = entry[l];
      const forms = typeof v === 'object' ? Object.values(v) : [v];
      assert.ok(forms.every((f) => typeof f === 'string'), `${key}.${l}`);
      if (!EMPTY_ALLOWED.has(key)) assert.ok(forms.every((f) => f.trim() !== ''), `${key}.${l} is empty`);
    }
    assert.equal(placeholders(entry.en), placeholders(entry.zh), `${key}: the same placeholders`);
    if (typeof entry.en === 'object') assert.deepEqual(Object.keys(entry.en).sort(), ['one', 'other'], key);
    assert.equal(typeof entry.zh, 'string', `${key}: Chinese has no plural forms`);
    assert.ok(!CJK.test(JSON.stringify(entry.en)), `${key}: English without CJK: ${JSON.stringify(entry.en)}`);
  }
  // the language names in the switch: each in its own language
  assert.deepEqual(LANG_NAMES, { zh: '中文', en: 'English' });
});

test('every key the pages ask for exists, and every text is used', () => {
  const used = new Set();
  const prefixes = new Set();
  for (const f of JS_FILES.filter((f) => f !== 'strings.js')) {
    const { code: src, literals } = scanJs(read(f));
    assert.doesNotMatch(src, /\bt\(\s*`/, `${f}: t() takes a literal key, not a template`);
    for (const m of src.matchAll(/\bt\(\s*(['"])(.*?)\1/g)) assert.ok(Object.hasOwn(STRINGS, m[2]), `${f}: t('${m[2]}') has no text`);
    for (const m of src.matchAll(/\bkeyed\(\s*(['"])(.*?)\1/g)) prefixes.add(m[2]);
    for (const lit of literals) if (Object.hasOwn(STRINGS, lit)) used.add(lit);
  }
  for (const f of HTML_FILES) {
    for (const el of elements(parseHtml(read(f)))) {
      for (const [name, key] of Object.entries(el.attrs)) {
        if (name !== 'data-i18n' && !name.startsWith('data-i18n-')) continue;
        assert.ok(Object.hasOwn(STRINGS, key), `${f}: [${name}="${key}"] has no text`);
        used.add(key);
      }
    }
  }
  const unused = Object.keys(STRINGS).filter((k) => !used.has(k) && ![...prefixes].some((p) => k.startsWith(`${p}.`)));
  assert.deepEqual(unused, [], 'texts nothing uses');
});

test('t(): the page language from <html data-lang>, placeholders, plurals, numbers, missing keys', () => {
  assert.equal(lang(), 'zh', 'node has no <html data-lang>: the language the pages are written in');
  assert.equal(locale(), 'zh-CN');
  assert.equal(t('time.minutesAgo', { n: 5 }), '5 分钟前');
  assert.equal(t('time.minutesAgo', { n: 1 }), '1 分钟前');
  inLang('en', () => {
    assert.equal(lang(), 'en');
    assert.equal(locale(), 'en');
    assert.equal(t('time.minutesAgo', { n: 1 }), '1 minute ago');
    assert.equal(t('time.minutesAgo', { n: 5 }), '5 minutes ago');
    assert.equal(t('time.minutesAgo', { n: 0 }), '0 minutes ago');
    assert.equal(t('unit.m', { n: 1234 }), '1,234 m', 'numbers are formatted for the language');
    assert.equal(t('unit.m', { n: '1234' }), '1234 m', 'strings pass through');
    assert.equal(t('unit.m'), '{n} m', 'a placeholder without a value stays');
  });
  assert.equal(lang(), 'zh', 'inLang restores the language');
  // a document that says English
  const saved = globalThis.document;
  globalThis.document = { documentElement: { dataset: { lang: 'en' } } };
  try {
    assert.equal(lang(), 'en');
    assert.equal(t('time.justNow'), 'just now');
    globalThis.document.documentElement.dataset.lang = 'fr';
    assert.equal(lang(), 'zh', 'an unknown data-lang: the authored language');
  } finally {
    globalThis.document = saved;
  }
  const before = missing.size;
  assert.equal(t('no.such.key'), 'no.such.key');
  assert.ok(missing.has('no.such.key'));
  assert.equal(missing.size, before + 1);
  assert.equal(t('toString'), 'toString', 'not a key: an Object.prototype name');
  missing.delete('no.such.key');
  missing.delete('toString');
});

test('keyed(): a table of codes read in the page language, like the plain object it replaces', () => {
  const table = keyed('prefs.lang', ['system']);
  assert.deepEqual(Object.keys(table), ['system']);
  assert.ok('system' in table);
  assert.ok(!('zh' in table));
  assert.equal(table.zh, undefined);
  assert.equal(table.system, '跟随系统');
  assert.deepEqual(Object.entries(table), [['system', '跟随系统']]);
  inLang('en', () => {
    assert.equal(table.system, 'System');
    assert.deepEqual(Object.values(table), ['System']);
  });
  assert.throws(() => {
    'use strict';
    table.system = 'x';
  }, TypeError);
  assert.equal(table.system, '跟随系统');
});

test('every keyed() table of the modules reads a text in both languages, never its key', async () => {
  // the modules node can import (viewer.js needs three.js; the page scripts need their page)
  const PAGES = ['boot.js', 'index.js', 'landmark.js', 'terms.js', 'viewer.js'];
  const MODULES = ['api.js', 'compass.js', 'framing.js', 'i18n.js', 'models.js', 'photoposes.js', 'poses.js', 'status.js',
    'strings.js', 'stylized.js', 'themes.js', 'ui.js'];
  assert.deepEqual([...PAGES, ...MODULES].sort(), JS_FILES, 'a new module: add it to MODULES (or PAGES)');
  globalThis.__STYLIZED_NO_AUTOSTART = true;
  for (const f of MODULES) await import(`../${f}`);
  for (const { prefix, codes, table } of keyedTables) {
    for (const l of ['zh', 'en']) {
      inLang(l, () => {
        for (const code of codes) {
          const v = table[code];
          assert.equal(typeof v, 'string', `${prefix}.${code} (${l})`);
          assert.notEqual(v, `${prefix}.${code}`, `${prefix}.${code} has no ${l} text`);
          assert.notEqual(v.trim(), '', `${prefix}.${code} (${l})`);
        }
      });
    }
  }
});

test('applyDom(): [data-i18n] text and [data-i18n-<attribute>] attributes, <html lang>, and the hidden English page shown', () => {
  const doc = installDom();
  try {
    const html = new FakeElement('html');
    const classes = new Set(['i18n-pending']);
    html.classList = { remove: (c) => classes.delete(c) };
    html.dataset.lang = 'en';
    doc.documentElement = html;
    const title = doc.createElement('title');
    title.setAttribute('data-i18n', 'terms.docTitle');
    title.textContent = STRINGS['terms.docTitle'].zh;
    doc.head.append(title);
    const meta = doc.createElement('meta');
    meta.setAttribute('content', STRINGS['terms.metaDescription'].zh);
    meta.setAttribute('data-i18n-content', 'terms.metaDescription');
    doc.head.append(meta);
    const link = doc.createElement('a');
    link.setAttribute('data-i18n', 'common.backToAll');
    link.setAttribute('data-i18n-aria-label', 'prefs.group');
    doc.body.append(link);
    applyDom(doc);
    assert.equal(title.textContent, 'Terms of Use · Crowd3D');
    assert.equal(meta.getAttribute('content'), STRINGS['terms.metaDescription'].en);
    assert.equal(link.textContent, '← All landmarks');
    assert.equal(link.getAttribute('aria-label'), 'Display');
    assert.equal(html.getAttribute('lang'), 'en');
    assert.ok(!classes.has('i18n-pending'));
    // Chinese: the text the HTML already has
    html.dataset.lang = 'zh';
    applyDom(doc);
    assert.equal(title.textContent, '用户协议 · Crowd3D 众拍建模');
    assert.equal(link.textContent, '← 全部地标');
    assert.equal(html.getAttribute('lang'), 'zh-CN');
  } finally {
    doc.restore();
  }
  applyDom(undefined); // no document: nothing to do
});

test('Intl helpers: numbers, dates and lists in the page language', () => {
  const d = new Date(2026, 9, 21, 9, 5);
  assert.equal(fmtDateTime(d), '10月21日 09:05');
  assert.equal(fmtNum(1234.5), '1,234.5');
  assert.equal(fmtList(['北面', '东面', '南面']), '北面、东面、南面');
  inLang('en', () => {
    assert.equal(fmtDateTime(d).replace(/ /g, ' '), 'Oct 21, 9:05 AM');
    assert.equal(fmtDateTime(new Date(2026, 9, 21, 14, 5)).replace(/ /g, ' '), 'Oct 21, 2:05 PM');
    assert.equal(fmtList(['north', 'east', 'south']), 'north, east, and south');
    assert.equal(fmtList(['north', 'east']), 'north and east');
  });
});

// ---------- Chinese outside strings.js: a ratchet ----------

// How many string literals (web/*.js, comments left out) and text runs or attribute values (web/*.html, outside
// [data-i18n], keyed attributes and [data-lang-only="zh"]) still hold CJK characters, per file. Steps 4 to 6 of D32 move
// them into strings.js: a count may only go down, and when it does, lower it here (a file at 0 leaves the map). Step 6
// deletes the map: then no file has any.
const LEGACY = {
  'api.js': 47,
  'compass.js': 18,
  'index.js': 28,
  'landmark.js': 80,
  'models.js': 5,
  'photoposes.js': 44,
  'poses.js': 39,
  'status.js': 17,
  'stylized.js': 82,
  'themes.js': 14,
  'viewer.js': 3,
  'index.html': 19,
  'landmark.html': 39,
  'stylized.html': 11,
};

test('Chinese written outside strings.js only goes down (LEGACY ratchet)', () => {
  const counts = {};
  for (const f of JS_FILES.filter((f) => f !== 'strings.js')) {
    const n = jsLiterals(read(f)).filter((s) => CJK.test(s)).length;
    if (n) counts[f] = n;
  }
  for (const f of HTML_FILES) {
    const n = htmlCjk(parseHtml(read(f)), f);
    if (n) counts[f] = n;
  }
  const problems = [];
  for (const f of [...new Set([...Object.keys(counts), ...Object.keys(LEGACY)])].sort()) {
    const now = counts[f] ?? 0;
    const allowed = LEGACY[f] ?? 0;
    if (now > allowed) problems.push(`${f}: ${now} texts with Chinese outside strings.js, more than ${allowed}: put new texts in strings.js`);
    if (now < allowed) problems.push(`${f}: down to ${now} texts with Chinese outside strings.js: lower LEGACY['${f}'] from ${allowed}`);
  }
  assert.deepEqual(problems, [], `the counts now: ${JSON.stringify(counts)}`);
});

test('the scanners: comments, regex literals, nested templates; HTML text, keyed elements and the Chinese agreement', () => {
  const src = [
    "// '注释' in a comment",
    '/* "注释" */ const a = \'甲\', b = "乙";',
    'const r = /["\']/g; const q = x / 2 / y;',
    'const s = `丙${n ? `丁` : \'\'}戊`;',
    'if (/张/.test(v)) return /[\'"]/;',
    "const url = 'http://example.com/a'; // 注释",
  ].join('\n');
  const lits = jsLiterals(src);
  assert.deepEqual(lits.filter((s) => CJK.test(s)), ['甲', '乙', '丁', '丙戊']);
  assert.ok(lits.includes('http://example.com/a'));
  const html = `<!doctype html><html><head><title data-i18n="brand.sub">众拍建模</title>
    <meta name="description" content="说明"><script>const x = '中';</script></head>
    <body><!-- 注释 --><p>你好 <b>世界</b></p><img alt="图" src="x.png">
    <p data-i18n-aria-label="prefs.group" aria-label="显示设置">x</p>
    <article data-lang-only="zh"><p>中文版</p></article><article data-lang-only="en"><p>English</p></article></body></html>`;
  assert.equal(htmlCjk(parseHtml(html), 'probe'), 4); // content, 你好, 世界, alt
  assert.throws(() => htmlCjk(parseHtml('<p data-i18n="brand.sub">别的</p>'), 'probe'), /zh text/);
  assert.throws(() => htmlCjk(parseHtml('<p data-i18n="brand.sub">众拍<b>建模</b></p>'), 'probe'), /text only/);
});

// ---------- the pages ----------

test('every page runs boot.js first (a classic script before any stylesheet, import map or module) and has the switch', () => {
  assert.ok(HTML_FILES.length >= 4);
  for (const f of HTML_FILES) {
    const all = [...elements(parseHtml(read(f)))];
    const scripts = all.filter((e) => e.tag === 'script');
    const boot = scripts[0];
    assert.equal(boot?.attrs.src, 'boot.js', `${f}: boot.js is the first script`);
    for (const a of ['type', 'defer', 'async']) assert.ok(!(a in boot.attrs), `${f}: boot.js is a classic, blocking script (${a})`);
    assert.equal(scripts.filter((s) => s.attrs.src === 'boot.js').length, 1, f);
    const first = all.findIndex(
      (e) => (e.tag === 'link' && /\bstylesheet\b/.test(e.attrs.rel ?? '')) || e.tag === 'style' || (e.tag === 'script' && e !== boot),
    );
    assert.ok(first < 0 || all.indexOf(boot) < first, `${f}: boot.js before any stylesheet, import map or module`);
    assert.equal(all.filter((e) => e.attrs.id === 'prefs').length, 1, `${f}: one #prefs`);
  }
});

test('the user agreement: two language versions with the same version and the same sections in the same order', () => {
  const all = [...elements(parseHtml(read('terms.html')))];
  const articles = all.filter((e) => e.tag === 'article');
  assert.deepEqual(articles.map((a) => a.attrs['data-lang-only']), ['zh', 'en']);
  assert.deepEqual(articles.map((a) => a.attrs.lang), ['zh-CN', 'en']);
  const [zh, en] = articles;
  assert.ok(zh.attrs['data-terms-version']);
  assert.equal(en.attrs['data-terms-version'], zh.attrs['data-terms-version']);
  const sections = (a) => [...elements(a)].filter((e) => e.tag === 'section').map((e) => e.attrs['data-section']);
  assert.ok(sections(zh).length >= 10);
  assert.deepEqual(sections(en), sections(zh));
  // style.css shows the version of the page language only
  const css = read('style.css');
  assert.match(css, /html\[data-lang="en"\] \[data-lang-only="zh"\],\s*html:not\(\[data-lang="en"\]\) \[data-lang-only="en"\] \{\s*display: none;/);
});
