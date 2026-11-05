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

const GLOBE_SVG =
  '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" ' +
  'stroke-linecap="round" stroke-linejoin="round" focusable="false"><circle cx="12" cy="12" r="9"/>' +
  '<path d="M3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9c-2.5-2.6-3.8-5.6-3.8-9S9.5 5.6 12 3z"/></svg>';

/**
 * The display settings (D32) in host (#prefs): a language select (跟随系统 / 中文 / English) that saves the choice and
 * reloads the page (web/boot.js window.crowd3dPrefs). Renders nothing and returns null without a host or without
 * boot.js (as in the node page tests); otherwise returns the group element.
 */
export function mountPrefs(host) {
  const prefs = globalThis.window?.crowd3dPrefs;
  if (!host || !prefs) return null;
  const choice = prefs.langChoice();
  const select = el(
    'select',
    { id: 'pref-lang', onchange: () => prefs.setLang(select.value) },
    el('option', { value: 'system', selected: choice === 'system', text: t('prefs.lang.system') }),
    el('option', { value: 'zh', lang: 'zh-CN', selected: choice === 'zh', text: LANG_NAMES.zh }),
    el('option', { value: 'en', lang: 'en', selected: choice === 'en', text: LANG_NAMES.en }),
  );
  select.value = choice;
  const group = el(
    'div',
    { class: 'prefs', role: 'group', 'aria-label': t('prefs.group') },
    el(
      'div',
      { class: 'pref' },
      el('label', { for: 'pref-lang', class: 'visually-hidden', text: t('prefs.lang') }),
      el('span', { class: 'pref-icon', 'aria-hidden': 'true', html: GLOBE_SVG }),
      select,
    ),
  );
  fill(host, group);
  return group;
}
