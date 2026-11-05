// Small DOM and formatting helpers shared by the pages. Text always goes in via
// textContent, so names, user ids and file names from the API are never parsed as HTML.
// The formatters write in the page language (web/i18n.js); mountPrefs draws the display settings (D32).

import { LANG_NAMES } from './strings.js';
import { fmtCount, fmtDateTime, fmtPercent as fmtFraction, t } from './i18n.js';

export { fmtCount };

/**
 * el('a', {class: 'x', href: '#', text: 'hi', onclick: fn, dataset: {k: 'v'}}, ...children)
 * Attribute values of null/undefined/false are skipped; true sets an empty attribute.
 */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'text') node.textContent = value;
    else if (key === 'html') node.innerHTML = value; // only for markup built locally (compass SVG)
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  node.append(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

/** A fraction, clamped to 0..1, as a whole percentage: '63%'. */
export function fmtPercent(p) {
  return fmtFraction(Math.max(0, Math.min(1, p || 0)));
}

/** Metres: one decimal below 10 m, else whole metres ('3.1 米' / '3.1 m', '43 米' / '43 m'). */
export function fmtMeters(m) {
  if (m === null || m === undefined) return '—';
  return t('unit.m', { n: m < 10 ? m.toFixed(1) : String(Math.round(m)) });
}

export function fmtDeg(d) {
  return d === null || d === undefined ? '—' : `${Math.round(d) % 360}°`;
}

/**
 * Unix seconds -> '刚刚' / '5 分钟前' / '3 小时前' / '10月21日 14:05'
 * (in English: 'just now' / '5 minutes ago' / '3 hours ago' / 'Oct 21, 2:05 PM').
 */
export function fmtTime(unix, now = Date.now() / 1000) {
  if (!unix) return '—';
  const s = Math.max(0, now - unix);
  if (s < 60) return t('time.justNow');
  if (s < 3600) return t('time.minutesAgo', { n: Math.floor(s / 60) });
  if (s < 86400) return t('time.hoursAgo', { n: Math.floor(s / 3600) });
  return fmtDateTime(new Date(unix * 1000));
}

export function progressBar(progress, label) {
  const pct = Math.round(Math.max(0, Math.min(1, progress || 0)) * 100);
  const bar = el('div', {
    class: 'progress',
    role: 'progressbar',
    'aria-valuemin': 0,
    'aria-valuemax': 100,
    'aria-valuenow': pct,
    'aria-label': label,
  });
  bar.append(el('div', { class: 'progress-fill', style: `width: ${pct}%` }));
  return bar;
}

/** Replace a container's children. */
export function fill(container, ...children) {
  container.replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
}

const ICON_ATTRS =
  'viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" ' +
  'stroke-linecap="round" stroke-linejoin="round" focusable="false"';
const GLOBE_SVG =
  `<svg ${ICON_ATTRS}><circle cx="12" cy="12" r="9"/>` +
  '<path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z"/></svg>';
// a circle, its right half filled: light and dark
const THEME_SVG = `<svg ${ICON_ATTRS}><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/></svg>`;

/** One setting of the group: a visually hidden label, an icon and a native select (autocomplete off, see mountPrefs). */
function prefSelect(id, label, icon, options, choice, onchange) {
  const select = el(
    'select',
    { id, autocomplete: 'off', onchange: () => onchange(select.value) },
    options.map(([value, text, lang]) => el('option', { value, lang, selected: choice === value, text })),
  );
  select.value = choice;
  const pref = el(
    'div',
    { class: 'pref' },
    el('label', { for: id, class: 'visually-hidden', text: label }),
    el('span', { class: 'pref-icon', 'aria-hidden': 'true', html: icon }),
    select,
  );
  return { pref, select };
}

/**
 * The display settings (D32) in host (#prefs), from web/boot.js window.crowd3dPrefs: a language select (跟随系统 /
 * 中文 / English) that saves the choice and reloads the page, and a theme select (跟随系统 / 浅色 / 深色) that saves
 * the choice and applies it at once. Renders nothing and returns null without a host or without boot.js (as in the
 * node page tests); otherwise returns the group element. The selects are autocomplete="off": a browser that restores
 * form values on Back would otherwise show the choice of the page left behind, not the stored one (a page restored
 * whole from the back/forward cache is reloaded by boot.js when the language changed meanwhile, and gets the stored
 * theme again). The theme select follows boot.js's 'crowd3d:theme' event, so a choice made in another tab shows.
 */
export function mountPrefs(host) {
  const prefs = globalThis.window?.crowd3dPrefs;
  if (!host || !prefs) return null;
  const lang = prefSelect(
    'pref-lang',
    t('prefs.lang'),
    GLOBE_SVG,
    [
      ['system', t('prefs.lang.system')],
      ['zh', LANG_NAMES.zh, 'zh-CN'],
      ['en', LANG_NAMES.en, 'en'],
    ],
    prefs.langChoice(),
    (choice) => prefs.setLang(choice),
  );
  const theme = prefSelect(
    'pref-theme',
    t('prefs.theme'),
    THEME_SVG,
    [
      ['system', t('prefs.theme.system')],
      ['light', t('prefs.theme.light')],
      ['dark', t('prefs.theme.dark')],
    ],
    prefs.themeChoice(),
    (choice) => prefs.setTheme(choice),
  );
  globalThis.window.addEventListener?.('crowd3d:theme', () => {
    theme.select.value = prefs.themeChoice();
  });
  const group = el('div', { class: 'prefs', role: 'group', 'aria-label': t('prefs.group') }, lang.pref, theme.pref);
  fill(host, group);
  return group;
}
