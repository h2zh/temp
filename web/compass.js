// Compass rose for landmark coverage. Pure: builds an SVG string, no DOM access,
// so it runs (and is unit-tested) under plain node.
//
// One wedge per sector, sector 0 centred on north at the top, indices clockwise,
// matching crowd3d/coverage.py. Colours come from CSS classes (is-locked,
// is-partial, is-face_ready) so light/dark themes are handled in style.css.
// Texts in the page language (web/strings.js compass.*): a sector has a prose name (东北 / north-east, inside
// sentences), a short one (东北 / NE, around the rose) and a facade name (东北立面 / North-east face).

import { fmtNum, keyed, locale, t } from './i18n.js';

export const SECTOR_LABELS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const SECTOR_NAMES = keyed('compass.sector', SECTOR_LABELS);
const SECTOR_SHORT = keyed('compass.short', SECTOR_LABELS);

export const STATE_TEXT = keyed('compass.state', ['locked', 'partial', 'face_ready']);

const CENTER = 100;
const FULL = { outer: 74, inner: 30, count: 46, check: 63, label: 88 };
const COMPACT = { outer: 96, inner: 34, check: 66 };

const own = (table, key) => typeof key === 'string' && Object.hasOwn(table, key);

/** The name of a sector label from the API inside a sentence ("NE" -> "东北" / "north-east"); unknown labels pass through. */
export function sectorName(label) {
  return own(SECTOR_NAMES, label) ? SECTOR_NAMES[label] : String(label);
}

/** The short name of a sector label ("NE" -> "东北" / "NE"); unknown labels pass through. */
export function sectorShort(label) {
  return own(SECTOR_SHORT, label) ? SECTOR_SHORT[label] : String(label);
}

/** A text with its first letter capitalised for the start of a line or title ("north-east" -> "North-east"). */
export function capitalize(text) {
  const s = String(text);
  return s.charAt(0).toLocaleUpperCase(locale()) + s.slice(1);
}

/** The name of a sector at the start of a line or title ("NE" -> "东北" / "North-east"); unknown labels pass through. */
export function sectorTitle(label) {
  return own(SECTOR_NAMES, label) ? capitalize(SECTOR_NAMES[label]) : String(label);
}

/** The facade of a sector ("NE" -> "东北立面" / "North-east face"); side: a name given instead of the label's. */
export function sectorFace(label, side = sectorTitle(label)) {
  return t('compass.face', { side });
}

/** Point at compass bearing `deg` (clockwise from north) and radius `r`, in SVG coordinates. */
export function polar(deg, r, c = CENTER) {
  const a = (deg * Math.PI) / 180;
  return [c + r * Math.sin(a), c - r * Math.cos(a)];
}

function n2(v) {
  return Math.round(v * 100) / 100;
}

function pt([x, y]) {
  return `${n2(x)} ${n2(y)}`;
}

export function escapeXml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
}

/** SVG path of an annulus wedge between bearings a0 and a1 (clockwise, a1 > a0). */
export function wedgePath(a0, a1, rInner, rOuter) {
  const large = a1 - a0 > 180 ? 1 : 0;
  return [
    `M ${pt(polar(a0, rOuter))}`,
    `A ${rOuter} ${rOuter} 0 ${large} 1 ${pt(polar(a1, rOuter))}`,
    `L ${pt(polar(a1, rInner))}`,
    `A ${rInner} ${rInner} 0 ${large} 0 ${pt(polar(a0, rInner))}`,
    'Z',
  ].join(' ');
}

function checkMark(x, y, s) {
  const d = `M ${pt([x - 0.5 * s, y])} L ${pt([x - 0.15 * s, y + 0.35 * s])} L ${pt([x + 0.5 * s, y - 0.35 * s])}`;
  return `<path class="compass-check" d="${d}" fill="none" stroke-width="${n2(s * 0.22)}" stroke-linecap="round" stroke-linejoin="round"/>`;
}

// A "!" badge drawn as shapes (no font needed) and without a nested <g>.
function failMark(x, y, s) {
  return (
    `<circle class="compass-fail-badge" cx="${n2(x)}" cy="${n2(y)}" r="${n2(s * 0.7)}"/>` +
    `<path class="compass-fail-mark" d="M ${pt([x, y - 0.38 * s])} L ${pt([x, y + 0.08 * s])}" fill="none" stroke-width="${n2(s * 0.2)}" stroke-linecap="round"/>` +
    `<circle class="compass-fail-dot" cx="${n2(x)}" cy="${n2(y + 0.36 * s)}" r="${n2(s * 0.11)}"/>`
  );
}

function centerOf(sector, i, n) {
  return Number.isFinite(sector.bearing_center_deg) ? sector.bearing_center_deg : (i * 360) / n;
}

/**
 * SVG markup for a list of SectorState objects ({index, label, count, state,
 * bearing_center_deg, has_face_model}).
 * Options: size (px), compact (no text, for cards), faceMin (photos a sector needs),
 * failed (sector indices whose newest facade job failed).
 */
export function compassSvg(sectors, { size = 240, compact = false, faceMin = null, failed = [] } = {}) {
  const list = Array.isArray(sectors) ? sectors : [];
  const n = list.length;
  const g = compact ? COMPACT : FULL;
  const cls = compact ? 'compass compass--compact' : 'compass';
  const open = (label) =>
    `<svg xmlns="http://www.w3.org/2000/svg" class="${cls}" viewBox="0 0 200 200" width="${size}" height="${size}" role="img" aria-label="${escapeXml(label)}">`;

  if (n === 0) {
    return `${open(t('compass.empty'))}<circle class="compass-wedge is-locked" cx="${CENTER}" cy="${CENTER}" r="${g.outer}"/></svg>`;
  }

  const width = 360 / n;
  const ready = list.filter((s) => s.state === 'face_ready').length;
  const total = list.reduce((sum, s) => sum + (s.count || 0), 0);
  const failedSet = new Set(failed);
  const parts = [open(t('compass.summary', { sides: n, ready, n: total }))];

  list.forEach((s, i) => {
    const mid = centerOf(s, i, n);
    const state = STATE_TEXT[s.state] ? s.state : 'locked';
    const failedHere = failedSet.has(s.index ?? i);
    const details = [
      faceMin ? t('compass.tip.countOf', { n: s.count, min: faceMin }) : t('compass.tip.count', { n: s.count }),
      STATE_TEXT[state],
    ];
    if (s.has_face_model) details.push(t('compass.tip.built'));
    if (failedHere) details.push(s.has_face_model ? t('compass.tip.updateFailed') : t('compass.faceFailed'));
    const tip = t('compass.tip', { side: sectorTitle(s.label), label: String(s.label), details: details.join(t('compass.tip.sep')) });
    parts.push(
      `<g class="compass-sector" data-index="${i}" data-state="${state}">`,
      `<title>${escapeXml(tip)}</title>`,
      `<path class="compass-wedge is-${state}" d="${wedgePath(mid - width / 2, mid + width / 2, g.inner, g.outer)}"/>`,
    );
    if (!compact && state !== 'locked') {
      const [x, y] = polar(mid, g.count);
      parts.push(
        `<text class="compass-count is-${state}" x="${n2(x)}" y="${n2(y)}" text-anchor="middle" dominant-baseline="central">${fmtNum(s.count)}</text>`,
      );
    }
    if (s.has_face_model) {
      const [x, y] = polar(mid, g.check);
      parts.push(checkMark(x, y, compact ? 20 : 11));
    } else if (failedHere) {
      const [x, y] = polar(mid, g.check);
      parts.push(failMark(x, y, compact ? 20 : 11));
    }
    parts.push('</g>');
  });

  parts.push(`<circle class="compass-core" cx="${CENTER}" cy="${CENTER}" r="${g.inner - 4}"/>`);
  if (compact) {
    // North tick on the core so the mini rose still reads as oriented.
    const [x, y] = polar(0, g.inner - 8);
    parts.push(`<path class="compass-north" d="M ${pt([x - 8, y + 12])} L ${pt([x, y - 4])} L ${pt([x + 8, y + 12])} Z"/>`);
  } else {
    parts.push(
      `<text class="compass-total" x="${CENTER}" y="${CENTER - 5}" text-anchor="middle" dominant-baseline="central">${fmtNum(total)}</text>`,
      `<text class="compass-total-unit" x="${CENTER}" y="${CENTER + 11}" text-anchor="middle" dominant-baseline="central">${escapeXml(t('compass.totalUnit', { n: total }))}</text>`,
    );
    list.forEach((s, i) => {
      const [x, y] = polar(centerOf(s, i, n), g.label);
      const north = i === 0 ? ' is-north' : '';
      parts.push(
        `<text class="compass-label${north}" x="${n2(x)}" y="${n2(y)}" text-anchor="middle" dominant-baseline="central">${escapeXml(sectorShort(s.label))}</text>`,
      );
    });
  }
  parts.push('</svg>');
  return parts.join('');
}
