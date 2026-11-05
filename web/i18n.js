// The page language (D32): t(key, params) looks a text up in web/strings.js for the language web/boot.js put on
// <html data-lang> ('zh' | 'en'; without it, 'zh', the language the pages are written in, which is also what the
// node tests see). applyDom() translates the static HTML ([data-i18n] and [data-i18n-<attribute>] elements);
// scripts call t() for the text they render. The language never changes while a page is open: switching reloads.
//
// Rules for the code that uses this (D32 steps 4 to 8 move the remaining Chinese text of the pages into strings.js):
// - No translatable string at module top level: a text looked up at import would stay in the language of that
//   moment (node tests switch the language after importing). Use a function, or keyed() for a table of codes.
// - Whole sentences as templates with {name} parameters, never fragments glued together with punctuation; word
//   order and punctuation differ between the languages.
// - No key built by concatenation outside keyed(): every key is a literal t('...') or a keyed() prefix, so the
//   tests can check that each key exists and each entry is used.
// - Numbers passed as params are formatted for the language (Intl.NumberFormat: 1,234); pass a string where the
//   digits must stay as they are (ids, years) or a fixed number of decimals matters (x.toFixed(1)).
// - Technical text stays untranslated: a job's raw error under the technical-details label, the theme validator's
//   messages, licence attributions. User content (landmark names, nicknames, file names) is never translated.
// - While extracting, the Chinese output stays byte-identical to what the page showed before.

import { STRINGS } from './strings.js';

export const LANGS = ['zh', 'en'];
const LOCALES = { zh: 'zh-CN', en: 'en' };

let override = null;

/** Keys asked for that strings.js does not have (each also shows as the key itself). */
export const missing = new Set();

/** Every keyed() table, for the tests: [{prefix, codes, table}]. */
export const keyedTables = [];

/** Tests only: force a language ('zh' | 'en'), or null to read <html data-lang> again. */
export function useLang(l) {
  override = LANGS.includes(l) ? l : null;
}

/** 'zh' | 'en'. */
export function lang() {
  if (override) return override;
  const l = globalThis.document?.documentElement?.dataset?.lang;
  return LANGS.includes(l) ? l : 'zh';
}

/** The BCP 47 locale of the page language, for Intl and <html lang>: 'zh-CN' | 'en'. */
export function locale() {
  return LOCALES[lang()];
}

function param(value) {
  return typeof value === 'number' && Number.isFinite(value) ? fmtNum(value) : String(value);
}

/** The text of a key in the page language, with its {name} placeholders filled from params. */
export function t(key, params = {}) {
  const entry = Object.hasOwn(STRINGS, key) ? STRINGS[key] : null;
  let value = entry?.[lang()];
  if (value && typeof value === 'object') {
    value = value[new Intl.PluralRules(locale()).select(Number(params.n))] ?? value.other; // { one, other }
  }
  if (typeof value !== 'string') {
    missing.add(key);
    return key;
  }
  return value.replace(/\{(\w+)\}/g, (m, name) => (Object.hasOwn(params, name) ? param(params[name]) : m));
}

/**
 * A table from codes to their texts under prefix ('reason' + 'no_gps' -> t('reason.no_gps')), read in the page
 * language when it is read. It works like the plain object it replaces: table[code], code in table,
 * Object.keys / values / entries; an unknown code gives undefined.
 */
export function keyed(prefix, codes) {
  const keys = Object.fromEntries(codes.map((code) => [code, `${prefix}.${code}`]));
  const table = new Proxy(keys, {
    get: (target, prop, receiver) =>
      typeof prop === 'string' && Object.hasOwn(target, prop) ? t(target[prop]) : Reflect.get(target, prop, receiver),
    set: () => false,
    deleteProperty: () => false,
    defineProperty: () => false,
  });
  keyedTables.push({ prefix, codes: [...codes], table });
  return table;
}

const ATTRIBUTES = ['aria-label', 'placeholder', 'title', 'content', 'alt'];

/**
 * Translates the static HTML: the text of every [data-i18n="key"] element and the attribute of every
 * [data-i18n-<attribute>="key"] element (aria-label, placeholder, title, content, alt); then sets <html lang> and
 * shows a page boot.js hid until now (html.i18n-pending). A [data-i18n] element holds text only.
 */
export function applyDom(doc = globalThis.document) {
  if (!doc) return;
  const roots = typeof doc.querySelectorAll === 'function' ? [doc] : [doc.head, doc.body].filter(Boolean);
  for (const root of roots) {
    for (const node of root.querySelectorAll('[data-i18n]')) node.textContent = t(node.getAttribute('data-i18n'));
    for (const name of ATTRIBUTES) {
      const attr = `data-i18n-${name}`;
      for (const node of root.querySelectorAll(`[${attr}]`)) node.setAttribute(name, t(node.getAttribute(attr)));
    }
  }
  const html = doc.documentElement;
  if (html) {
    html.setAttribute('lang', locale());
    html.classList?.remove('i18n-pending');
  }
}

/** A number in the page language (Intl.NumberFormat options as given). */
export function fmtNum(n, options = {}) {
  return new Intl.NumberFormat(locale(), options).format(n);
}

/** A count, with grouping ('1,234'); null and undefined count as 0. */
export function fmtCount(n) {
  return fmtNum(n ?? 0);
}

/** A fraction (0.42) as a whole percentage ('42%'), rounded half up like Math.round. */
export function fmtPercent(fraction) {
  return fmtNum(Math.round(fraction * 100) / 100, { style: 'percent', maximumFractionDigits: 0 });
}

const DATE_TIME = {
  zh: { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }, // 10月21日 09:05
  en: { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }, // Oct 21, 9:05 AM
};

/**
 * A date and time of day in local time, without the year: '10月21日 09:05' / 'Oct 21, 9:05 AM'. The time and its
 * AM/PM never break apart: the space before the day period is a no-break space (Chromium writes a plain one there,
 * newer ICU a narrow no-break one).
 */
export function fmtDateTime(date) {
  return new Intl.DateTimeFormat(locale(), DATE_TIME[lang()])
    .formatToParts(date)
    .map((p, i, parts) => (p.type === 'literal' && parts[i + 1]?.type === 'dayPeriod' ? p.value.replace(/\s/gu, ' ') : p.value))
    .join('');
}

// the enumeration comma (Intl.ListFormat would end a Chinese list with 和); formatting, not text (i18n.test.mjs NOT_TEXT)
const ZH_LIST_SEPARATOR = '、';

/** A list of items: 'a、b、c' / 'a, b, and c'. */
export function fmtList(items) {
  const list = [...items].map(String);
  if (lang() === 'zh') return list.join(ZH_LIST_SEPARATOR);
  return new Intl.ListFormat(locale(), { style: 'long', type: 'conjunction' }).format(list);
}
