// Compass rose for landmark coverage. Pure: builds an SVG string, no DOM access,
// so it runs (and is unit-tested) under plain node.
//
// One wedge per sector, sector 0 centred on north at the top, indices clockwise,
// matching crowd3d/coverage.py. Colours come from CSS classes (is-locked,
// is-partial, is-face_ready) so light/dark themes are handled in style.css.

const SECTOR_NAMES = { N: '北', NE: '东北', E: '东', SE: '东南', S: '南', SW: '西南', W: '西', NW: '西北' };

const STATE_TEXT = { locked: '暂无照片', partial: '照片不足', face_ready: '照片已够' };

const CENTER = 100;
const FULL = { outer: 74, inner: 30, count: 46, check: 63, label: 88 };
const COMPACT = { outer: 96, inner: 34, check: 66 };

/** Chinese name for a sector label from the API ("NE" -> "东北"); unknown labels pass through. */
export function sectorName(label) {
  return SECTOR_NAMES[label] ?? String(label);
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

function centerOf(sector, i, n) {
  return Number.isFinite(sector.bearing_center_deg) ? sector.bearing_center_deg : (i * 360) / n;
}

/**
 * SVG markup for a list of SectorState objects ({index, label, count, state,
 * bearing_center_deg, has_face_model}).
 * Options: size (px), compact (no text, for cards), faceMin (photos a sector needs).
 */
export function compassSvg(sectors, { size = 240, compact = false, faceMin = null } = {}) {
  const list = Array.isArray(sectors) ? sectors : [];
  const n = list.length;
  const g = compact ? COMPACT : FULL;
  const cls = compact ? 'compass compass--compact' : 'compass';
  const open = (label) =>
    `<svg xmlns="http://www.w3.org/2000/svg" class="${cls}" viewBox="0 0 200 200" width="${size}" height="${size}" role="img" aria-label="${escapeXml(label)}">`;

  if (n === 0) {
    return `${open('暂无覆盖数据')}<circle class="compass-wedge is-locked" cx="${CENTER}" cy="${CENTER}" r="${g.outer}"/></svg>`;
  }

  const width = 360 / n;
  const ready = list.filter((s) => s.state === 'face_ready').length;
  const total = list.reduce((sum, s) => sum + (s.count || 0), 0);
  const parts = [open(`覆盖罗盘：${n} 个方向中 ${ready} 个照片已够，共 ${total} 张照片`)];

  list.forEach((s, i) => {
    const mid = centerOf(s, i, n);
    const state = STATE_TEXT[s.state] ? s.state : 'locked';
    const name = sectorName(s.label);
    let tip = `${name}（${s.label}）：${s.count} 张${faceMin ? ` / ${faceMin}` : ''}，${STATE_TEXT[state]}`;
    if (s.has_face_model) tip += '，已生成立面模型';
    parts.push(
      `<g class="compass-sector" data-index="${i}" data-state="${state}">`,
      `<title>${escapeXml(tip)}</title>`,
      `<path class="compass-wedge is-${state}" d="${wedgePath(mid - width / 2, mid + width / 2, g.inner, g.outer)}"/>`,
    );
    if (!compact && state !== 'locked') {
      const [x, y] = polar(mid, g.count);
      parts.push(
        `<text class="compass-count is-${state}" x="${n2(x)}" y="${n2(y)}" text-anchor="middle" dominant-baseline="central">${s.count}</text>`,
      );
    }
    if (s.has_face_model) {
      const [x, y] = polar(mid, g.check);
      parts.push(checkMark(x, y, compact ? 20 : 11));
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
      `<text class="compass-total" x="${CENTER}" y="${CENTER - 5}" text-anchor="middle" dominant-baseline="central">${total}</text>`,
      `<text class="compass-total-unit" x="${CENTER}" y="${CENTER + 11}" text-anchor="middle" dominant-baseline="central">张</text>`,
    );
    list.forEach((s, i) => {
      const [x, y] = polar(centerOf(s, i, n), g.label);
      const north = i === 0 ? ' is-north' : '';
      parts.push(
        `<text class="compass-label${north}" x="${n2(x)}" y="${n2(y)}" text-anchor="middle" dominant-baseline="central">${escapeXml(sectorName(s.label))}</text>`,
      );
    });
  }
  parts.push('</svg>');
  return parts.join('');
}
