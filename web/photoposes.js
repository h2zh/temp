// Public photo poses on the clay model (D30): where each photo whose uploader agreed was taken and which way it looked,
// drawn on web/stylized.html from the landmark's public poses (GET /api/landmarks/{id}/poses, format
// crowd3d.photo-poses version 2), loaded with stylized.html?model=...&poses=<same-origin URL> (models.js stylizedView
// adds it). The document's parser and its texts are web/poses.js's, shared with the landmark page; the layer's own texts
// are web/strings.js pp.* (D32: Chinese and English).
//
// The server sends only accepted photos uploaded under the user agreement and not withdrawn (D32), with a determined
// pose, already rounded (whole metres, whole degrees, fields of view to whole degrees) and keyed by a salted public id.
// Nothing here rounds again, fills in a missing value, or shows a time or a nickname. The photo itself is shown only on
// request, in the comparison below. In the landmark ENU frame, which is the GLB's:
//
//   src  sfm  = a camera of a published SfM model whose frame is fixed  -> frustum, orange
//        loc  = localized against a published model                     -> frustum, purple
//        exif = the phone's own GPS and true-north compass, never checked against the model
//               (「手机 GPS + 罗盘（未经模型核对）」)                   -> heading wedge and dot on the ground (p[2] is null:
//               the height was not measured)
//   A frustum needs the measured height, pitch, roll, field of view and aspect. An sfm or loc row without one of them is
//   drawn as a ground wedge too, never as a frustum held level, or opened to a field of view, by assumption.
//   sp / sy: the row's 2-sigma ranges (metres, degrees). The position disc (radius sp) is drawn on the ground for every
//   row with sp > DISC_SP_M, so a glyph never looks more precise than its range; other rows show it when picked.
//   「从这张照片的角度看」 (D32, web/compare.js) shows the photo and the model seen from its pose together: the stage is
//   resized to the model pane (the photo's aspect) and the 3-D camera is placed at the photo's pose with its field of
//   view, the photo beside it (side by side or stacked) or over it (an opacity slider); ‹ › step around the landmark,
//   photo and model together. Only for a row drawn as a frustum (all five measured); any other row's card offers its
//   photo alone (看照片, photo-only: the honest label and why it cannot be compared; viewFromBlocked: a code,
//   BLOCKED_TEXT its text). The photo is GET .../poses/{public id}/photo, fetched when shown (compare.js photoUrl).
//   The legend's counts are the server's (counts: shared, shown, per source, not_determined). The legend and the card
//   quote each source's measured coverage as the server sends it (uncertainty.measured_coverage; poses.js).
//
// Clutter: glyphs keep a minimum on-screen size; glyphs closer than CLUSTER_PX on screen merge into a count bubble with
// a ring of heading ticks; a click on a bubble flies in. Hover shows a one-line tip, a click the card.
//
// Ported by hand from the photo-pose layer of #87 without its GPS-only dots, view cells, thumbnails, times, own-photo
// marks and "approx" poses. The pure helpers are exported for node tests; three.js comes in through createPhotoLayer's
// env.

import {
  PUBLIC_ID_RE,
  cardActions,
  compareBanner,
  compareLayout,
  comparePrefs,
  compareSearch,
  createCompareView,
  noCompareText,
  parseCompareParams,
  photoUrl,
  readComparePrefs,
  writeComparePrefs,
} from './compare.js';
import { fmtCount, keyed, lang, locale, t } from './i18n.js';
import {
  SOURCE_HINT,
  SOURCES,
  countsText,
  coverageText,
  headingText,
  parsePoses,
  positionText,
  sourceCoverageText,
} from './poses.js';
import { adoptCss } from './ui.js';

export const STYLE = {
  sfm: { color: '#e8590c', glyph: 'frustum', order: 0 },
  loc: { color: '#7b2cbf', glyph: 'frustum', order: 1 },
  exif: { color: '#f0a400', glyph: 'wedge', order: 2 },
};
// the position disc is drawn for every row whose 2-sigma range is wider than this (metres): an unsnapped SfM camera
// (at least 16 m public), any localization into such a model and every EXIF row; a snapped model's cameras show theirs
// when picked
export const DISC_SP_M = 10;
export const WEDGE_MIN_HALF_DEG = 8; // a heading wedge is never drawn narrower than +-8 deg (its sy when wider)
export const CLUSTER_PX = 26;
export const CLUSTER_MIN = 3;

// The sources as the layer names them (the legend shows a label without its hint, so the EXIF label keeps the owner's
// full wording in English too; the landmark page's narrow table uses poses.js SOURCE_TEXT).
export const SOURCE_LABEL = keyed('pp.source', SOURCES);

// Why "view from this photo" is off (viewFromBlocked's codes).
export const BLOCKED = ['exif', 'pose', 'fov'];
export const BLOCKED_TEXT = keyed('pp.blocked', BLOCKED);

const isInt = (v) => Number.isInteger(v);
const finite = (v) => typeof v === 'number' && Number.isFinite(v);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const rad = (d) => (d * Math.PI) / 180;

/**
 * Options from location.search: poses=<same-origin URL>, photos=0 (start hidden), photo=<public id> (select),
 * view=<public id> (open "view from this photo": the comparison, or a blocked row's photo alone), cmp=pair|overlay and
 * opacity=0..100 (the comparison's mode and the photo's opacity in overlay mode, over the stored choice; clamped, in
 * steps of 5; anything else: null, the stored or default one), err=1 (every range drawn), cluster=0, fit=model (the
 * first view frames the model only; default: the model and the photos).
 */
export function parsePoseParams(search, origin = 'http://localhost') {
  const params = new URLSearchParams(search);
  let url = params.get('poses');
  if (url) {
    try {
      const u = new URL(url, origin + '/');
      url = u.origin === new URL(origin).origin ? u.pathname + u.search : null;
    } catch {
      url = null;
    }
  }
  const id = (k) => (PUBLIC_ID_RE.test(params.get(k) ?? '') ? params.get(k) : null);
  return {
    url: url || null,
    show: params.get('photos') !== '0',
    select: id('photo'),
    view: id('view'),
    allErrors: params.get('err') === '1',
    cluster: params.get('cluster') !== '0',
    fit: params.get('fit') === 'model' ? 'model' : 'photos',
    ...parseCompareParams(params),
  };
}

/**
 * The layer's input from a poses document, or null when it is not one (poses.js parsePoses: version 2 only; a row with
 * an unknown source, or without a whole-metre position, a heading and its ranges, is dropped). The values are the
 * server's, unchanged; an optional value that is not a plausible one (pitch beyond +-90 deg, a field of view outside
 * 1-150 deg, ...) becomes null, never a default. dropped: rows the parser could not place.
 */
export function layerInput(doc) {
  const parsed = parsePoses(doc);
  if (!parsed) return null;
  const photos = parsed.photos.map((r) => ({
    id: r.id,
    src: r.src,
    p: [r.p[0], r.p[1], r.p[2]],
    yaw: r.yaw,
    pitch: isInt(r.pitch) && Math.abs(r.pitch) <= 90 ? r.pitch : null,
    roll: isInt(r.roll) && Math.abs(r.roll) <= 180 ? r.roll : null,
    hfov: isInt(r.hfov) && r.hfov >= 1 && r.hfov <= 150 ? r.hfov : null,
    aspect: finite(r.aspect) && r.aspect >= 0.2 && r.aspect <= 5 ? r.aspect : null,
    sp: r.sp,
    sy: r.sy,
  }));
  const c = parsed.counts && typeof parsed.counts === 'object' && !Array.isArray(parsed.counts) ? parsed.counts : {};
  const counts = Object.fromEntries(
    ['photos', 'shared', 'shown', 'not_determined', ...SOURCES].map((k) => [k, isInt(c[k]) && c[k] >= 0 ? c[k] : null]),
  );
  return {
    photos,
    counts,
    dropped: doc.photos.length - photos.length,
    uncertainty: parsed.uncertainty && typeof parsed.uncertainty === 'object' ? parsed.uncertainty : null,
    rounding: parsed.rounding && typeof parsed.rounding === 'object' ? parsed.rounding : null,
    asOf: finite(parsed.as_of) ? parsed.as_of : null,
  };
}

/**
 * The glyph of a row: a frustum (sfm / loc with a measured height, pitch, roll, field of view and aspect: the same rows
 * "view from this photo" accepts) or a heading wedge on the ground.
 */
export function glyphOf(ph) {
  return STYLE[ph.src].glyph === 'frustum' && viewFromBlocked(ph) === null ? 'frustum' : 'wedge';
}

/** Whether a row's position disc is drawn: always beyond DISC_SP_M, else when the row is picked or every range is on. */
export function showsDisc(ph, { selected = false, all = false } = {}) {
  return ph.sp > DISC_SP_M || selected || all;
}

/**
 * Why "view from this photo" is off for a row, or null when it is on: it needs the measured camera height, pitch, roll,
 * field of view and aspect. 'exif': an EXIF row never has them (GPS and compass give a rough position and heading
 * only); 'pose': no measured height, pitch or roll; 'fov': no measured field of view or aspect. BLOCKED_TEXT[code] says
 * it in words.
 */
export function viewFromBlocked(ph) {
  if (ph.src === 'exif') return 'exif';
  if (ph.p[2] === null || ph.pitch === null || ph.roll === null) return 'pose';
  if (ph.hfov === null || ph.aspect === null) return 'fov';
  return null;
}

/** ENU (x east, y north, z up) -> three.js (x east, y up, z south), the GLB's mapping. */
export function enuToThree(p) {
  return [p[0], p[2] ?? 0, -p[1]];
}

/**
 * Camera basis in three.js axes for a pose: forward (viewing direction), right and up. yaw = compass bearing of the
 * viewing direction, pitch + up, roll + when the image's right side is raised (crowd3d/photoposes.py's convention).
 */
export function viewBasis(yaw, pitch = 0, roll = 0) {
  const y = rad(yaw);
  const p = rad(pitch ?? 0);
  const q = rad(roll ?? 0);
  const f = [Math.sin(y) * Math.cos(p), Math.cos(y) * Math.cos(p), Math.sin(p)]; // ENU
  let rl = [f[1], -f[0], 0]; // f x up
  const n = Math.hypot(rl[0], rl[1]) || 1;
  rl = rl.map((v) => v / n);
  const ul = [rl[1] * f[2] - rl[2] * f[1], rl[2] * f[0] - rl[0] * f[2], rl[0] * f[1] - rl[1] * f[0]]; // rl x f
  const right = rl.map((v, i) => Math.cos(q) * v + Math.sin(q) * ul[i]);
  const up = [right[1] * f[2] - right[2] * f[1], right[2] * f[0] - right[0] * f[2], right[0] * f[1] - right[1] * f[0]];
  return { forward: enuToThree(f), right: enuToThree(right), up: enuToThree(up) };
}

/** Vertical field of view (deg) from the horizontal one and the aspect (width / height). */
export function vfovOf(hfov, aspect) {
  return (2 * Math.atan(Math.tan(rad(hfov) / 2) / aspect) * 180) / Math.PI;
}

/** The 3-D camera's vertical fov and the photo's frame (fractions of the viewport) to show a photo in a viewport. */
export function photoFrame(hfov, aspect, viewW, viewH) {
  const va = viewW / viewH;
  const vfovPhoto = vfovOf(hfov, aspect);
  if (va >= aspect) {
    // viewport wider than the photo: heights match
    const w = aspect / va;
    return { fov: vfovPhoto, frame: { x: (1 - w) / 2, y: 0, w, h: 1 } };
  }
  const h = va / aspect;
  return { fov: (2 * Math.atan(Math.tan(rad(vfovPhoto) / 2) / h) * 180) / Math.PI, frame: { x: 0, y: (1 - h) / 2, w: 1, h } };
}

/**
 * Where "view from this photo" puts the 3-D camera for a row in a viewport of W x H pixels (three.js axes): the
 * row's own position, viewing direction and field of view; null when it is off (viewFromBlocked).
 */
export function viewPose(ph, W, H) {
  if (viewFromBlocked(ph)) return null;
  const b = viewBasis(ph.yaw, ph.pitch, ph.roll);
  const pf = photoFrame(ph.hfov, ph.aspect, W, H);
  return { position: enuToThree(ph.p), forward: b.forward, up: b.up, right: b.right, fov: pf.fov, frame: pf.frame };
}

/** "位置 ±16 米，朝向 ±21°" / "position ±16 m, heading ±21°" */
export function errorText(ph) {
  return t('pp.error', { sp: ph.sp, sy: ph.sy });
}

/** How the server rounded (doc.rounding), in words: position, angles and field of view. */
export function roundingText(rounding) {
  const m = isInt(rounding?.position_m) ? rounding.position_m : 1;
  const d = isInt(rounding?.angle_deg) ? rounding.angle_deg : 1;
  const f = isInt(rounding?.fov_deg) ? rounding.fov_deg : 1;
  return t('pp.rounding', { m, d, f });
}

/** The card's last line and the legend's: the rounding, what is never shown, and how the photo itself is published. */
export function privacyText(rounding) {
  return t('pp.privacy', { rounding: roundingText(rounding) });
}

/** Lines of the card for one row ({k, v}): source, position, heading, field of view, range and how it held, rounding. */
export function describePhoto(ph, { uncertainty = null, rounding = null } = {}) {
  const [, , u] = ph.p;
  const at = { position: positionText(ph.p), sp: ph.sp };
  let pos = t('pp.card.pos', at);
  if (u === null) pos = t('pp.card.posNoHeight', at);
  else if (Math.abs(u) >= 3) pos = t('pp.card.posHeight', { ...at, height: u });
  const dir = { heading: headingText(ph.yaw), sy: ph.sy };
  let heading = t('pp.card.dir', dir);
  if (ph.pitch === null) heading = t('pp.card.dirNoPitch', dir);
  else if (Math.abs(ph.pitch) >= 3) heading = t(ph.pitch > 0 ? 'pp.card.dirUp' : 'pp.card.dirDown', { ...dir, pitch: Math.abs(ph.pitch) });
  const sigmas = finite(uncertainty?.sigmas) ? uncertainty.sigmas : 2;
  return [
    { k: 'title', v: t('pp.card.title') },
    { k: 'src', v: t('pp.card.source', { source: SOURCE_LABEL[ph.src], hint: SOURCE_HINT[ph.src] }) },
    { k: 'pos', v: pos },
    { k: 'dir', v: heading },
    { k: 'fov', v: ph.hfov === null ? t('pp.card.fovNone') : t('pp.card.fov', { hfov: ph.hfov }) },
    { k: 'range', v: t('pp.card.range', { sigmas, coverage: sourceCoverageText(uncertainty, ph.src) }) },
    { k: 'privacy', v: privacyText(rounding) },
  ];
}

/** The legend's range of the drawn uncertainties of one source: "±16–41 米 / ±21–31°" ("" without rows). */
export function errorRange(photos, src) {
  const rows = photos.filter((ph) => ph.src === src);
  if (!rows.length) return '';
  const span = (vals) => [Math.min(...vals), Math.max(...vals)];
  const [plo, phi] = span(rows.map((ph) => ph.sp));
  const [ylo, yhi] = span(rows.map((ph) => ph.sy));
  const position = plo === phi ? t('poses.range.position', { n: plo }) : t('pp.rangeSpan', { lo: plo, hi: phi });
  const heading = ylo === yhi ? `±${ylo}°` : `±${ylo}–${yhi}°`;
  return `${position} / ${heading}`;
}

/** The one-line tip of a row under the pointer: its source, heading and ranges. */
export function tipText(ph) {
  return t('pp.tip', { source: SOURCE_LABEL[ph.src], heading: headingText(ph.yaw), error: errorText(ph) });
}

/** A cluster bubble's tooltip from its counts per source ({src: n}, in the order given). */
export function bubbleTitle(bySource) {
  const counts = Object.entries(bySource).map(([s, n]) => t('pp.bubble.count', { source: SOURCE_LABEL[s], n }));
  return t('pp.bubble.title', { counts: counts.join(' · ') });
}

/** Counts per source of the drawn rows. */
export function countBySource(photos) {
  const out = Object.fromEntries(SOURCES.map((k) => [k, 0]));
  for (const ph of photos) out[ph.src] += 1;
  return out;
}

/**
 * The legend's numbers: the server's counts (over the landmark's photos up to the snapshot), else, where the server
 * sent none, the drawn rows; drawn: the rows this page could place.
 */
export function legendTotals(input) {
  const drawn = countBySource(input.photos);
  const c = input.counts;
  const bySource = Object.fromEntries(SOURCES.map((k) => [k, c[k] ?? drawn[k]]));
  return { drawn, bySource, counts: c, shown: c.shown ?? input.photos.length };
}

const AS_OF_EN = { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' };

/** "2026-11-04 06:12" / "Nov 4, 2026, 6:12 AM" (local time) from a Unix time in seconds. */
export function asOfText(seconds) {
  if (!finite(seconds)) return '';
  const d = new Date(seconds * 1000);
  if (lang() === 'en') return new Intl.DateTimeFormat(locale(), AS_OF_EN).format(d);
  const z = (v) => String(v).padStart(2, '0');
  return `${d.getFullYear()}-${z(d.getMonth() + 1)}-${z(d.getDate())} ${z(d.getHours())}:${z(d.getMinutes())}`;
}

/**
 * Greedy screen-space clusters. items: [{i, x, y, prio}] (pixels; lower prio = seed first); items within radius px of
 * a seed join it. Returns [{seed, members: [i...], x, y}] (members include the seed).
 */
export function clusterScreen(items, radius = CLUSTER_PX) {
  const order = [...items].sort((a, b) => a.prio - b.prio || a.i - b.i);
  const clusters = [];
  const r2 = radius * radius;
  for (const it of order) {
    let best = null;
    let bd = r2;
    for (const c of clusters) {
      const dd = (c.x - it.x) ** 2 + (c.y - it.y) ** 2;
      if (dd <= bd) {
        bd = dd;
        best = c;
      }
    }
    if (best) best.members.push(it.i);
    else clusters.push({ seed: it.i, members: [it.i], x: it.x, y: it.y });
  }
  return clusters;
}

/** The rows used to frame the first view: those within the q-quantile of distance from the centre. */
export function framingSet(photos, q = 0.9) {
  if (!photos.length) return [];
  const d = photos.map((ph) => Math.hypot(ph.p[0], ph.p[1])).sort((a, b) => a - b);
  const lim = d[Math.min(d.length - 1, Math.floor(q * (d.length - 1)))] * 1.0001;
  return photos.filter((ph) => Math.hypot(ph.p[0], ph.p[1]) <= lim);
}

/**
 * A heading wedge on the ground in three.js coordinates (x east, z = -north), apex (cx, cz): yaw +- half degrees,
 * radius len, at height y. Returns {tri: [x, y, z, ...] triangles, line: segment pairs (arc + both sides)}.
 */
export function wedgeGeometry(cx, cz, yaw, half, len, y = 0.06) {
  const tri = [];
  const line = [];
  const steps = Math.max(4, Math.ceil(half / 4));
  const at = (a) => [cx + Math.sin(a) * len, y, cz - Math.cos(a) * len];
  let prev = null;
  for (let k = 0; k <= steps; k++) {
    const pt = at(rad(yaw - half + (2 * half * k) / steps));
    if (prev) {
      tri.push(cx, y, cz, ...prev, ...pt);
      line.push(...prev, ...pt);
    }
    prev = pt;
  }
  line.push(cx, y, cz, ...at(rad(yaw - half)));
  line.push(cx, y, cz, ...at(rad(yaw + half)));
  return { tri, line };
}

/** A disc on the ground (three.js axes): triangles and its rim as segment pairs. */
export function discGeometry(cx, cz, r, y = 0.04, steps = 48) {
  const tri = [];
  const rim = [];
  for (let k = 0; k < steps; k++) {
    const a0 = (2 * Math.PI * k) / steps;
    const a1 = (2 * Math.PI * (k + 1)) / steps;
    const p0 = [cx + Math.cos(a0) * r, y, cz + Math.sin(a0) * r];
    const p1 = [cx + Math.cos(a1) * r, y, cz + Math.sin(a1) * r];
    tri.push(cx, y, cz, ...p0, ...p1);
    rim.push(...p0, ...p1);
  }
  return { tri, rim };
}

/** The legend's text lines (plain text; the page escapes them). */
export function legendNotes(input) {
  const notes = [];
  if (input.photos.some((ph) => ph.sp > DISC_SP_M)) notes.push(t('pp.legend.disc', { n: DISC_SP_M }));
  const cov = coverageText(input.uncertainty, input.photos, SOURCE_LABEL);
  if (cov) notes.push(cov);
  notes.push(privacyText(input.rounding));
  if (input.asOf !== null) notes.push(t('pp.legend.asOf', { time: asOfText(input.asOf) }));
  if (input.dropped > 0) notes.push(t('pp.legend.dropped', { n: input.dropped }));
  return notes;
}

/**
 * The legend panel's label and markup (a <details>: its title, the count line, a row per source present with its
 * filter box, swatch, label, range and count, the notes, and the two switches). allErrors, shown: the switches' state.
 */
export function legendMarkup(input, { allErrors = false, shown = true } = {}) {
  const tot = legendTotals(input);
  const swatch = (k) => `<span class="pp-sw pp-sw-${k} ${STYLE[k].glyph}"></span>`;
  const count = (k) => (tot.bySource[k] !== tot.drawn[k]
    ? t('pp.legend.countDrawn', { n: tot.bySource[k], drawn: tot.drawn[k] }) : fmtCount(tot.bySource[k]));
  const present = SOURCES.filter((k) => tot.bySource[k] || tot.drawn[k]);
  const countLine = input.counts.shared !== null && input.counts.photos !== null
    ? countsText(input.counts) : t('pp.legend.shown', { n: tot.shown });
  const html = `<summary><h2>${escapeHtml(t('pp.legend.title'))}</h2></summary>`
    + `<p class="pp-count">${escapeHtml(countLine)}</p>`
    + present.map((k) => `<label><input type="checkbox" data-src="${k}" checked ${tot.drawn[k] ? '' : 'disabled'}>${swatch(k)}`
      + `${escapeHtml(SOURCE_LABEL[k])}${errorRange(input.photos, k) ? `<span class="pp-err">${escapeHtml(errorRange(input.photos, k))}</span>` : ''}`
      + `<span class="pp-n">${escapeHtml(count(k))}</span></label>`).join('')
    + legendNotes(input).map((line) => `<p class="pp-note">${escapeHtml(line)}</p>`).join('')
    + `<label><input type="checkbox" data-err ${allErrors ? 'checked' : ''}>${escapeHtml(t('pp.legend.allRanges'))}</label>`
    + `<label><input type="checkbox" data-show ${shown ? 'checked' : ''}>${escapeHtml(t('pp.legend.show'))}</label>`;
  return { label: t('pp.legend.label'), html };
}

// ------------------------------------------------------------------ browser part

// The panels (legend, card, tip, buttons) follow the light/dark choice through stylized.html's --panel-* tokens (the
// fallbacks are the light values); the marks drawn over the model (.pp-bubble, .pp-north, .pp-ring) belong to the art
// theme like the model itself and keep --ink in both modes (D32). The comparison's own rules are web/compare.js's.
// The legend panel and the card share one column on the right (.pp-side, the card at its foot): on a wide screen the
// panel gives up height to the card and scrolls, so the two never overlap, whatever the language; on a phone, or a
// screen too low for both, the panel makes way while the card is open.
const CSS = `
.pp-side{position:fixed;z-index:3;right:max(12px,env(safe-area-inset-right));top:max(12px,env(safe-area-inset-top));
  bottom:max(12px,env(safe-area-inset-bottom));width:min(320px,calc(100vw - 24px));display:flex;flex-direction:column;
  align-items:flex-end;gap:12px;pointer-events:none}
.pp-side>*{pointer-events:auto}
.pp-panel{box-sizing:border-box;flex:0 1 auto;min-height:0;width:min(300px,100%);overflow:auto;padding:10px 12px;border-radius:16px;
  background:var(--panel,#fff);color:var(--panel-fg,#2d2a32);border:3px solid var(--panel-line,#2d2a32);
  box-shadow:0 4px 0 var(--panel-line,#2d2a32);font-size:13px}
.pp-panel h2{margin:0 0 6px;font-size:15px;display:inline}
.pp-panel summary{cursor:pointer;margin-bottom:4px}
.pp-panel:not([open]) summary{margin:0}
body.pp-card-open .pp-panel{display:none}
.pp-panel label{display:flex;align-items:center;gap:6px;margin:2px 0;cursor:pointer;font-weight:700}
.pp-panel label input{margin:0}
.pp-sw{display:inline-block;flex:none;width:18px;height:12px;border-radius:3px;border:2px solid var(--panel-line,#2d2a32)}
.pp-sw.frustum{clip-path:polygon(0 0,100% 20%,100% 80%,0 100%)}
.pp-sw.wedge{border-radius:2px 12px 12px 2px;opacity:.85}
.pp-n{margin-left:auto;font-variant-numeric:tabular-nums}
.pp-err{margin-left:6px;color:var(--panel-soft,#5b5763);font-weight:400;font-size:11px;white-space:nowrap}
.pp-note{margin:4px 0 0;color:var(--panel-soft,#5b5763);font-size:11px;line-height:1.35}
.pp-count{margin:2px 0 4px;font-size:12px}
.pp-bubble{pointer-events:auto;cursor:pointer;position:relative;width:46px;height:46px;transform:translateY(-4px)}
.pp-bubble svg{position:absolute;inset:0;overflow:visible}
.pp-bubble b{position:absolute;inset:11px;border-radius:50%;background:#fff;border:2.5px solid var(--ink,#2d2a32);
  display:flex;align-items:center;justify-content:center;font:800 12px/1 system-ui,sans-serif;color:var(--ink,#2d2a32)}
.pp-bubble.occluded{opacity:.38}
.pp-tip{position:fixed;pointer-events:none;padding:3px 8px;border-radius:999px;background:var(--panel-chip,#fff);color:var(--panel-fg,#2d2a32);
  border:2px solid var(--panel-line,#2d2a32);
  font-size:12px;font-weight:700;white-space:nowrap;z-index:4;transform:translate(12px,-120%)}
.pp-card{box-sizing:border-box;flex:none;margin-top:auto;width:100%;max-height:100%;overflow:auto;padding:10px 12px;border-radius:14px;
  background:var(--panel,#fff);color:var(--panel-fg,#2d2a32);border:3px solid var(--panel-line,#2d2a32);
  box-shadow:0 4px 0 var(--panel-line,#2d2a32);font-size:13px}
.pp-card p{margin:2px 0}.pp-card .t{font-weight:800}.pp-card .s{font-size:11.5px;color:var(--panel-soft,#5b5763)}
.pp-card button{font:inherit;font-weight:800;padding:4px 10px;border-radius:999px;border:2.5px solid var(--panel-line,#2d2a32);
  background:var(--panel-chip,#fff);color:var(--panel-fg,#2d2a32);cursor:pointer;margin:6px 6px 0 0}
.pp-card button.go{background:var(--panel-go,#ffe8cc)}
.pp-ring{pointer-events:none;font:700 11px/1 system-ui,sans-serif;color:#4b5a44;background:rgba(255,255,255,.85);padding:1px 5px;border-radius:999px}
.pp-north{pointer-events:none;font:900 14px/1 system-ui,sans-serif;color:#fff;background:var(--ink,#2d2a32);padding:3px 6px;border-radius:6px}
body.pp-viewing #panel,body.pp-viewing .pp-panel,body.pp-viewing #pick{display:none}
body.no-ui .pp-panel,body.no-ui .pp-card{display:none}
@media (min-width:521px) and (min-height:481px){body.pp-card-open .pp-panel{display:block}}
@media (max-width:520px){.pp-side{left:12px;right:12px;width:auto;justify-content:flex-end;align-items:stretch}
  .pp-panel{width:auto;max-height:45vh}}
`;
// A legend swatch takes its source's colour from a rule (.pp-sw-sfm, ...), not from a style attribute: the pages'
// Content-Security-Policy (crowd3d/csp.py) allows none.
const SWATCH_CSS = SOURCES.map((k) => `.pp-sw-${k}{background:${STYLE[k].color}}`).join('\n');

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

/**
 * The photo layer. env: {THREE, scene, camera, controls, renderer, CSS2DObject, LineSegments2, LineSegmentsGeometry,
 * LineMaterial, resolution, look, drawn, render, state, footR, stage, resize}; stage: the element the renderer fills
 * (resized to the model pane while comparing), resize(): the viewer's own resize (renderer, labels, outline
 * resolutions, camera aspect; it calls the layer's resize()). input: layerInput() of the document; opts:
 * parsePoseParams(). Returns {update(), fitPoints(), handleTap(event) -> bool, resize(), select(id), viewFrom(id),
 * exitView(ms), setAllErrors(on), setClustering(on), setShown(on)}. env.state.compare: {id, mode, photo} while a
 * photo is shown (null otherwise); env.state.posesViewReady: the camera is placed and the photo loaded or failed.
 */
export function createPhotoLayer(env, input, opts) {
  const { THREE, scene, camera, controls, renderer, CSS2DObject } = env;
  const { LineSegments2, LineSegmentsGeometry, LineMaterial } = env;
  const photos = input.photos;
  const n = photos.length;
  const ink = env.look?.ink ?? '#2d2a32';
  adoptCss(document, CSS + SWATCH_CSS);

  const lineMats = [];
  const lineMat = (params) => {
    const m = new LineMaterial({ ...params, resolution: env.resolution });
    lineMats.push(m);
    return m;
  };
  const group = new THREE.Group();
  group.name = 'photo-poses';
  scene.add(group);
  const glyph = photos.map(glyphOf);
  const pos = photos.map((ph) => new THREE.Vector3(...enuToThree(ph.p)));
  const basis = photos.map((ph, i) => (glyph[i] === 'frustum' ? viewBasis(ph.yaw, ph.pitch, ph.roll) : viewBasis(ph.yaw)));
  const colour = photos.map((ph) => new THREE.Color(STYLE[ph.src].color));
  const visible = new Array(n).fill(true); // per source filter
  const clustered = new Array(n).fill(false);
  const enabled = Object.fromEntries(SOURCES.map((k) => [k, true]));
  let shown = opts.show;
  let allErrors = opts.allErrors;
  let clustering = opts.cluster;
  let selected = -1;
  let hover = -1;
  let viewing = null;

  // sizes: a frustum's depth in metres (by the building's size), never under MIN_PX on screen
  const D0 = clamp(0.06 * env.footR, 1.2, 3.5);
  const MIN_PX = 15;
  const DOT_PX = 4;

  // ---- frustums (glyphOf): translucent sides, brighter image plane, ink edges
  const frustumIdx = photos.map((ph, i) => (glyph[i] === 'frustum' ? i : -1)).filter((i) => i >= 0);
  const pyr = new THREE.BufferGeometry();
  pyr.setAttribute('position', new THREE.Float32BufferAttribute([
    0, 0, 0, -1, -1, -1, 1, -1, -1, 0, 0, 0, 1, -1, -1, 1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, -1, 1, -1, -1, -1, -1,
  ], 3));
  const plane = new THREE.BufferGeometry();
  plane.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1], 3));
  const sideMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false });
  const planeMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.8, side: THREE.DoubleSide });
  const sides = new THREE.InstancedMesh(pyr, sideMat, Math.max(1, frustumIdx.length));
  const planes = new THREE.InstancedMesh(plane, planeMat, Math.max(1, frustumIdx.length));
  for (const m of [sides, planes]) {
    m.frustumCulled = false;
    m.count = frustumIdx.length;
    frustumIdx.forEach((i, k) => m.setColorAt(k, colour[i]));
    group.add(m);
  }
  planes.renderOrder = 2;
  sides.renderOrder = 3;
  // edges: 8 per frustum + an "up" tick (2) so roll reads; per-segment colour (ink, white for the selected one)
  const SEG = 10;
  const edgeGeo = new LineSegmentsGeometry();
  const edgePos = new Float32Array(Math.max(1, frustumIdx.length) * SEG * 6);
  const edgeCol = new Float32Array(Math.max(1, frustumIdx.length) * SEG * 6);
  edgeGeo.setPositions(edgePos);
  edgeGeo.setColors(edgeCol);
  const edges = new LineSegments2(edgeGeo, lineMat({ vertexColors: true, linewidth: 1.4 }));
  edges.frustumCulled = false;
  group.add(edges);
  const inkC = new THREE.Color(ink);
  const whiteC = new THREE.Color('#ffffff');
  // image-plane corners and the "up" tick in the frustum's local frame (looking down -z, half-extent 1 at depth 1)
  const LOCAL = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-0.45, 1.08, -1], [0.45, 1.08, -1], [0, 1.55, -1]];
  const local = LOCAL.map(() => new THREE.Vector3());

  // ---- dots (every row): a small disc facing the camera at the camera position (an EXIF row's on the ground); a white
  // halo for the selected one
  const dots = new THREE.InstancedMesh(new THREE.CircleGeometry(1, 12), new THREE.MeshBasicMaterial(), Math.max(1, n));
  dots.frustumCulled = false;
  dots.count = n;
  photos.forEach((ph, i) => dots.setColorAt(i, colour[i]));
  group.add(dots);
  const halo = new THREE.InstancedMesh(new THREE.RingGeometry(1.35, 1.9, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, depthTest: false, transparent: true }), 1);
  halo.frustumCulled = false;
  halo.count = 0;
  halo.renderOrder = 5;
  group.add(halo);
  // ground shadow dots + stalks for elevated vantage points (> 4 m: towers, terraces)
  const shadow = new THREE.InstancedMesh(new THREE.CircleGeometry(1, 16), new THREE.MeshBasicMaterial({ color: ink, transparent: true, opacity: 0.22, depthWrite: false }), Math.max(1, n));
  shadow.frustumCulled = false;
  shadow.count = n;
  group.add(shadow);
  const elevated = photos.map((ph, i) => (ph.p[2] !== null && ph.p[2] > 4 ? i : -1)).filter((i) => i >= 0);
  let stalks = null;
  if (elevated.length) {
    const g = new LineSegmentsGeometry().setPositions(elevated.flatMap((i) => [pos[i].x, 0.05, pos[i].z, pos[i].x, pos[i].y, pos[i].z]));
    stalks = new LineSegments2(g, lineMat({ color: ink, linewidth: 1, dashed: true, dashSize: 1.2, gapSize: 0.8, transparent: true, opacity: 0.6 }));
    stalks.computeLineDistances();
    group.add(stalks);
  }

  // ---- heading wedges on the ground in the source's colour (EXIF rows, and sfm / loc rows that are not frustums):
  // yaw +- max(sy, WEDGE_MIN_HALF_DEG); rebuilt when the set changes
  const wedgeIdx = photos.map((ph, i) => (glyph[i] === 'wedge' ? i : -1)).filter((i) => i >= 0);
  const wedgeLen = (i) => clamp(0.4 * Math.hypot(photos[i].p[0], photos[i].p[1]), 6, 22);
  const wedges = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.3, depthWrite: false, side: THREE.DoubleSide }));
  wedges.renderOrder = 1;
  group.add(wedges);
  const wedgeEdgeMat = lineMat({ vertexColors: true, linewidth: 1.6 });
  const edgeColour = photos.map((ph) => new THREE.Color(STYLE[ph.src].color).multiplyScalar(0.72));
  let wedgeEdges = null;
  const fan = (i, half, len, y = 0.06) => wedgeGeometry(pos[i].x, pos[i].z, photos[i].yaw, half, len, y);
  const repeatColour = (c, nVerts) => Array.from({ length: nVerts }, () => [c.r, c.g, c.b]).flat();
  function rebuildWedges() {
    const tri = [];
    const triCol = [];
    const line = [];
    const lineCol = [];
    for (const i of wedgeIdx) {
      if (!visible[i] || clustered[i] || i === viewingIdx()) continue;
      const f = fan(i, Math.max(WEDGE_MIN_HALF_DEG, photos[i].sy), wedgeLen(i));
      tri.push(...f.tri);
      triCol.push(...repeatColour(colour[i], f.tri.length / 3));
      line.push(...f.line);
      lineCol.push(...repeatColour(edgeColour[i], f.line.length / 3));
    }
    wedges.geometry.dispose();
    wedges.geometry = new THREE.BufferGeometry();
    wedges.geometry.setAttribute('position', new THREE.Float32BufferAttribute(tri, 3));
    wedges.geometry.setAttribute('color', new THREE.Float32BufferAttribute(triCol, 3));
    if (wedgeEdges) {
      group.remove(wedgeEdges);
      wedgeEdges.geometry.dispose();
      wedgeEdges = null;
    }
    if (line.length) {
      wedgeEdges = new LineSegments2(new LineSegmentsGeometry().setPositions(line).setColors(lineCol), wedgeEdgeMat);
      group.add(wedgeEdges);
    }
  }

  // ---- uncertainty: the position disc (radius sp) of every row with sp > DISC_SP_M, of the selected row, or of all
  // (err=1). The discs are filled as one union (the stencil lets each pixel be drawn once, so overlapping discs do not
  // pile up into an opaque blot), each in its source's colour; the selected row's disc gets a dashed rim. The
  // selected frustum also gets its heading range (yaw +- sy) on the ground.
  const discMat = new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide,
    stencilWrite: true, stencilRef: 1, stencilFunc: THREE.NotEqualStencilFunc, stencilZPass: THREE.ReplaceStencilOp,
  });
  const discs = new THREE.Mesh(new THREE.BufferGeometry(), discMat);
  discs.renderOrder = 1;
  group.add(discs);
  const rimMat = lineMat({ color: ink, linewidth: 1.4, dashed: true, dashSize: 1.2, gapSize: 0.9 });
  let rims = null;
  const errWedges = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, depthWrite: false, side: THREE.DoubleSide }));
  group.add(errWedges);
  function rebuildErrors() {
    const tri = [];
    const col = [];
    const rim = [];
    const wtri = [];
    // the selected row first, then the narrowest ranges: where discs overlap, the first one drawn keeps its colour, so
    // a glyph's own disc shows around it and the wide EXIF discs fill what is left
    const bySp = [...photos.keys()].sort((a, b) => photos[a].sp - photos[b].sp || a - b);
    const order = selected >= 0 ? [selected, ...bySp.filter((i) => i !== selected)] : bySp;
    for (const i of order) {
      if (!visible[i] || i === viewingIdx()) continue;
      if (!showsDisc(photos[i], { selected: i === selected, all: allErrors })) continue;
      const d = discGeometry(pos[i].x, pos[i].z, Math.max(0.6, photos[i].sp));
      tri.push(...d.tri);
      col.push(...repeatColour(colour[i], d.tri.length / 3));
      if (i === selected) {
        rim.push(...d.rim);
        if (glyph[i] === 'frustum' && photos[i].sy >= 2) wtri.push(...fan(i, photos[i].sy, wedgeLen(i), 0.05).tri);
      }
    }
    discs.geometry.dispose();
    discs.geometry = new THREE.BufferGeometry();
    discs.geometry.setAttribute('position', new THREE.Float32BufferAttribute(tri, 3));
    discs.geometry.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    errWedges.geometry.dispose();
    errWedges.geometry = new THREE.BufferGeometry();
    errWedges.geometry.setAttribute('position', new THREE.Float32BufferAttribute(wtri, 3));
    if (rims) {
      group.remove(rims);
      rims.geometry.dispose();
      rims = null;
    }
    if (rim.length) {
      rims = new LineSegments2(new LineSegmentsGeometry().setPositions(rim), rimMat);
      rims.computeLineDistances();
      group.add(rims);
    }
  }

  // ---- sight line of the selected frustum: from the camera along its view to the first wall it hits
  const sightMat = lineMat({ color: '#ffffff', linewidth: 2.2, dashed: true, dashSize: 1.0, gapSize: 0.7 });
  let sight = null;
  const ray = new THREE.Raycaster();
  function rebuildSight() {
    if (sight) {
      group.remove(sight);
      sight.geometry.dispose();
      sight = null;
    }
    if (selected < 0 || glyph[selected] !== 'frustum' || viewingIdx() >= 0) return;
    ray.set(pos[selected], new THREE.Vector3(...basis[selected].forward));
    ray.far = 2000;
    const hit = ray.intersectObjects(env.drawn, false)[0];
    if (!hit) return;
    const a = pos[selected];
    const b = hit.point;
    sight = new LineSegments2(new LineSegmentsGeometry().setPositions([a.x, a.y, a.z, b.x, b.y, b.z]), sightMat);
    sight.computeLineDistances();
    sight.renderOrder = 6;
    group.add(sight);
  }

  // ---- ground: a light plaza under the photos, distance rings and a north mark
  const fit = framingSet(photos);
  const dist0 = (ph) => Math.hypot(ph.p[0], ph.p[1]);
  const far = Math.max(env.footR * 1.4, ...photos.map((ph) => dist0(ph) + Math.min(ph.sp, 60))) * 1.06;
  const plazaR = Math.min(far, Math.max(env.footR * 1.4, ...fit.map((ph) => dist0(ph) + Math.min(ph.sp, 60))) * 1.08);
  const plaza = new THREE.Mesh(new THREE.CircleGeometry(plazaR, 96), new THREE.MeshBasicMaterial({ color: new THREE.Color(env.look?.ground?.color ?? '#cfeab8').lerp(new THREE.Color('#ffffff'), 0.45), transparent: true, opacity: 0.7, depthWrite: false }));
  plaza.rotation.x = -Math.PI / 2;
  plaza.position.y = -0.06;
  plaza.renderOrder = -1;
  group.add(plaza);
  const ringStep = plazaR > 260 ? 100 : plazaR > 110 ? 50 : plazaR > 45 ? 20 : 10;
  const ringSegs = [];
  for (let r = ringStep; r < plazaR; r += ringStep) {
    if (r < env.footR * 0.9) continue;
    for (let k = 0; k < 96; k++) {
      const a0 = (2 * Math.PI * k) / 96;
      const a1 = (2 * Math.PI * (k + 1)) / 96;
      ringSegs.push(Math.cos(a0) * r, 0.02, Math.sin(a0) * r, Math.cos(a1) * r, 0.02, Math.sin(a1) * r);
    }
    const el = document.createElement('div');
    el.className = 'pp-ring';
    el.textContent = t('unit.m', { n: r });
    const lab = new CSS2DObject(el);
    lab.position.set(r * Math.SQRT1_2, 0.1, r * Math.SQRT1_2); // south-east of the centre
    group.add(lab);
  }
  if (ringSegs.length) {
    const rl = new LineSegments2(new LineSegmentsGeometry().setPositions(ringSegs), lineMat({ color: '#7d9a6b', linewidth: 1, dashed: true, dashSize: 2, gapSize: 2, transparent: true, opacity: 0.7 }));
    rl.computeLineDistances();
    group.add(rl);
  }
  const northEl = document.createElement('div');
  northEl.className = 'pp-north';
  northEl.textContent = t('pp.north');
  const north = new CSS2DObject(northEl);
  north.position.set(0, 0.1, -plazaR);
  group.add(north);

  // ---- bubbles (clusters) and tip
  const bubblePool = [];
  function bubble(k) {
    while (bubblePool.length <= k) {
      const el = document.createElement('div');
      el.className = 'pp-bubble';
      el.innerHTML = '<svg width="46" height="46"></svg><b></b>';
      const obj = new CSS2DObject(el);
      obj.userData.members = [];
      el.addEventListener('pointerup', (e) => {
        e.stopPropagation();
        flyToMembers(obj.userData.members);
      });
      el.addEventListener('pointerdown', (e) => e.stopPropagation());
      group.add(obj);
      bubblePool.push(obj);
    }
    return bubblePool[k];
  }
  const tip = document.createElement('div');
  tip.className = 'pp-tip';
  tip.hidden = true;
  document.body.append(tip);

  // ---- legend panel
  const panel = document.createElement('details');
  panel.className = 'pp-panel';
  panel.open = !(window.matchMedia && window.matchMedia('(max-width: 520px)').matches);
  const legend = legendMarkup(input, { allErrors, shown });
  panel.setAttribute('aria-label', legend.label);
  panel.innerHTML = legend.html;
  const side = document.createElement('div');
  side.className = 'pp-side';
  side.append(panel);
  document.body.append(side);
  panel.addEventListener('change', (e) => {
    const box = e.target;
    if (box.dataset.src) {
      enabled[box.dataset.src] = box.checked;
      for (let i = 0; i < n; i++) visible[i] = enabled[photos[i].src];
      if (selected >= 0 && !visible[selected]) select(-1);
    } else if ('err' in box.dataset) allErrors = box.checked;
    else if ('show' in box.dataset) shown = box.checked;
    dirty = true;
    env.render();
  });

  // ---- details card
  const card = document.createElement('aside');
  card.className = 'pp-card';
  card.hidden = true;
  side.append(card);
  const setCard = (open) => {
    card.hidden = !open;
    document.body.classList.toggle('pp-card-open', open);
  };

  // ---- per-frame update
  const v = new THREE.Vector3();
  const m4 = new THREE.Matrix4();
  const sc = new THREE.Vector3();
  const groundQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
  const screen = new Array(n);
  let dirty = true;
  let lastClusterKey = '';
  let lastUpdateMs = 0;
  let clusters = [];
  const basisQuat = basis.map((b) => {
    const mm = new THREE.Matrix4().makeBasis(new THREE.Vector3(...b.right), new THREE.Vector3(...b.up), new THREE.Vector3(...b.forward).negate());
    return new THREE.Quaternion().setFromRotationMatrix(mm);
  });
  // the row compared with the model (its own glyph is not drawn then); a photo-only row leaves the overview as it is
  function viewingIdx() {
    return viewing && !viewing.photoOnly ? viewing.i : -1;
  }
  let occlusionTimer = null;
  const occRay = new THREE.Raycaster();
  function checkOcclusion() {
    for (const b of bubblePool) {
      if (!b.visible) continue;
      const to = b.position.clone().sub(camera.position);
      const d = to.length();
      occRay.set(camera.position, to.normalize());
      occRay.far = d - 0.5;
      b.element.classList.toggle('occluded', occRay.intersectObjects(env.drawn, false).length > 0);
    }
  }

  function update() {
    const t0 = performance.now();
    // while comparing nothing of the layer is drawn: the model is seen alone from the photo's pose
    const comparing = viewingIdx() >= 0;
    group.visible = shown && !comparing;
    // CSS2DRenderer only looks at each label's own flag, not its parents': ring labels and the north mark
    for (const c of group.children) if (c.isCSS2DObject && !bubblePool.includes(c)) c.visible = group.visible;
    const W = renderer.domElement.clientWidth || 1;
    const H = renderer.domElement.clientHeight || 1;
    if (!shown || comparing) {
      for (const b of bubblePool) b.visible = false;
      lastUpdateMs = performance.now() - t0;
      return;
    }
    const tanHalf = Math.tan(rad(camera.fov) / 2);
    // screen positions
    const items = [];
    for (let i = 0; i < n; i++) {
      v.copy(pos[i]).project(camera);
      const onScreen = v.z < 1 && Math.abs(v.x) < 1.2 && Math.abs(v.y) < 1.2;
      screen[i] = { x: (v.x + 1) * 0.5 * W, y: (1 - v.y) * 0.5 * H, on: onScreen };
      if (!visible[i]) continue;
      // the selected row never merges into a bubble
      if (onScreen && i !== selected && clustering) items.push({ i, x: screen[i].x, y: screen[i].y, prio: STYLE[photos[i].src].order });
    }
    clusters = clusterScreen(items).filter((c) => c.members.length >= CLUSTER_MIN);
    clustered.fill(false);
    for (const c of clusters) for (const i of c.members) clustered[i] = true;
    const key = clusters.map((c) => c.members.join(',')).join('|') + `/${selected}/${allErrors}/${visible.join('')}`;
    if (key !== lastClusterKey || dirty) {
      rebuildWedges();
      rebuildErrors();
      rebuildSight();
      lastClusterKey = key;
      dirty = false;
    }
    // glyphs: visible instances are packed to the front and .count / instanceCount cut there
    const camPos = camera.position;
    let m = 0;
    for (const i of frustumIdx) {
      if (!visible[i] || clustered[i]) continue;
      const d = pos[i].distanceTo(camPos);
      const perPx = (2 * d * tanHalf) / H;
      const big = i === selected || i === hover ? 1.5 : 1;
      const depth = Math.max(D0, MIN_PX * perPx) * big;
      // the image plane keeps the photo's field of view and aspect; very wide or tall frames are scaled down to a
      // half-extent of 1 so one wide-angle photo does not cover its neighbours
      const a0 = Math.tan(rad(photos[i].hfov) / 2);
      const b0 = a0 / photos[i].aspect;
      const shrink = Math.min(1, 1 / Math.max(a0, b0));
      sc.set(depth * a0 * shrink, depth * b0 * shrink, depth);
      m4.compose(pos[i], basisQuat[i], sc);
      sides.setMatrixAt(m, m4);
      planes.setMatrixAt(m, m4);
      sides.setColorAt(m, colour[i]);
      planes.setColorAt(m, colour[i]);
      for (let c = 0; c < 7; c++) local[c].fromArray(LOCAL[c]).applyMatrix4(m4);
      const apex = pos[i];
      const [c0, c1, c2, c3, t0b, t1b, t2b] = local;
      const segs = [apex, c0, apex, c1, apex, c2, apex, c3, c0, c1, c1, c2, c2, c3, c3, c0, t0b, t2b, t2b, t1b];
      const col = i === selected ? whiteC : inkC;
      for (let s2 = 0; s2 < SEG * 2; s2++) {
        const o = m * SEG * 6 + s2 * 3;
        edgePos[o] = segs[s2].x;
        edgePos[o + 1] = segs[s2].y;
        edgePos[o + 2] = segs[s2].z;
        edgeCol[o] = col.r;
        edgeCol[o + 1] = col.g;
        edgeCol[o + 2] = col.b;
      }
      m += 1;
    }
    sides.count = planes.count = m;
    sides.instanceMatrix.needsUpdate = planes.instanceMatrix.needsUpdate = true;
    if (sides.instanceColor) sides.instanceColor.needsUpdate = true;
    if (planes.instanceColor) planes.instanceColor.needsUpdate = true;
    edgeGeo.instanceCount = m * SEG;
    const ia = edgeGeo.attributes.instanceStart.data;
    ia.array.set(edgePos);
    ia.needsUpdate = true;
    const ic = edgeGeo.attributes.instanceColorStart.data;
    ic.array.set(edgeCol);
    ic.needsUpdate = true;
    // dots: flat discs facing the camera; shadows on the ground; a halo for the selected row
    let nd = 0;
    halo.count = 0;
    for (let i = 0; i < n; i++) {
      if (!visible[i] || clustered[i]) continue;
      const d = pos[i].distanceTo(camPos);
      const perPx = (2 * d * tanHalf) / H;
      const big = i === selected || i === hover ? 1.6 : 1;
      const r = Math.max(0.22, DOT_PX * perPx) * big;
      m4.compose(pos[i], camera.quaternion, sc.set(r, r, r));
      dots.setMatrixAt(nd, m4);
      dots.setColorAt(nd, colour[i]);
      if (i === selected) {
        halo.setMatrixAt(0, m4);
        halo.count = 1;
      }
      const sr = Math.max(0.3, 3 * perPx);
      m4.compose(v.set(pos[i].x, 0.03, pos[i].z), groundQ, sc.set(sr, sr, sr));
      shadow.setMatrixAt(nd, m4);
      nd += 1;
    }
    dots.count = shadow.count = nd;
    dots.instanceMatrix.needsUpdate = halo.instanceMatrix.needsUpdate = shadow.instanceMatrix.needsUpdate = true;
    if (dots.instanceColor) dots.instanceColor.needsUpdate = true;
    // bubbles: count + a ring of heading ticks (screen-space directions of the members' views)
    clusters.forEach((c, k) => {
      const b = bubble(k);
      b.visible = true;
      b.userData.members = c.members;
      let cx = 0;
      let cy = 0;
      let cz = 0;
      for (const i of c.members) {
        cx += pos[i].x;
        cy += pos[i].y;
        cz += pos[i].z;
      }
      const mm = c.members.length;
      b.position.set(cx / mm, cy / mm, cz / mm);
      const ticks = [];
      const byClass = {};
      for (const i of c.members) {
        byClass[photos[i].src] = (byClass[photos[i].src] ?? 0) + 1;
        const f = basis[i].forward;
        v.set(pos[i].x + f[0] * 5, pos[i].y + f[1] * 5, pos[i].z + f[2] * 5).project(camera);
        const tx = (v.x + 1) * 0.5 * W - screen[i].x;
        const ty = (1 - v.y) * 0.5 * H - screen[i].y;
        const L = Math.hypot(tx, ty);
        if (L < 1e-3) continue;
        const ux = tx / L;
        const uy = ty / L;
        ticks.push(`<line x1="${(23 + ux * 12).toFixed(1)}" y1="${(23 + uy * 12).toFixed(1)}" x2="${(23 + ux * 22).toFixed(1)}" y2="${(23 + uy * 22).toFixed(1)}" stroke="${STYLE[photos[i].src].color}" stroke-width="2.6" stroke-linecap="round"/>`);
      }
      const main = Object.entries(byClass).sort((p, r) => r[1] - p[1])[0][0];
      const el = b.element;
      const sig = `${mm}|${main}|${ticks.join('')}`;
      if (el.dataset.sig !== sig) {
        el.querySelector('svg').innerHTML = `<circle cx="23" cy="23" r="12" fill="none" stroke="${ink}" stroke-width="5"/>` + ticks.join('');
        const lab = el.querySelector('b');
        lab.textContent = String(mm);
        lab.style.borderColor = STYLE[main].color;
        el.title = bubbleTitle(byClass);
        el.dataset.sig = sig;
      }
    });
    for (let k = clusters.length; k < bubblePool.length; k++) bubblePool[k].visible = false;
    // bubbles are DOM (always on top): once the camera rests, the ones behind the building are faded
    clearTimeout(occlusionTimer);
    occlusionTimer = setTimeout(checkOcclusion, 200);
    lastUpdateMs = performance.now() - t0;
    env.state.poses = {
      photos: n, frustums: m, discs: photos.filter((ph, i) => visible[i] && showsDisc(ph, { selected: i === selected, all: allErrors })).length,
      clusters: clusters.length, clustered: clustered.filter(Boolean).length, updateMs: Math.round(lastUpdateMs * 100) / 100,
      selected: selected >= 0 ? photos[selected].id : null, viewing: viewing ? photos[viewing.i].id : null,
    };
  }

  // ---- picking (screen space: no ray casts)
  function nearest(x, y, maxPx = 14) {
    let best = -1;
    let bd = maxPx * maxPx;
    for (let i = 0; i < n; i++) {
      if (!visible[i] || clustered[i] || !screen[i]?.on) continue;
      const dd = (screen[i].x - x) ** 2 + (screen[i].y - y) ** 2;
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    }
    return best;
  }
  renderer.domElement.addEventListener('pointermove', (e) => {
    if (!shown || viewing) return;
    const r = renderer.domElement.getBoundingClientRect();
    const i = nearest(e.clientX - r.left, e.clientY - r.top);
    if (i !== hover) {
      hover = i;
      env.render();
    }
    if (i >= 0) {
      const ph = photos[i];
      tip.textContent = tipText(ph);
      tip.style.left = `${e.clientX}px`;
      tip.style.top = `${e.clientY}px`;
      tip.hidden = false;
    } else tip.hidden = true;
  });

  function handleTap(e) {
    if (viewing) return true; // a tap on the model pane while comparing picks nothing
    if (!shown) return false;
    const r = renderer.domElement.getBoundingClientRect();
    const i = nearest(e.clientX - r.left, e.clientY - r.top);
    if (i < 0) {
      if (selected >= 0) {
        select(-1);
        return true;
      }
      return false;
    }
    select(i);
    return true;
  }

  function select(i) {
    selected = typeof i === 'string' ? photos.findIndex((ph) => ph.id === i) : i;
    dirty = true;
    if (selected < 0) {
      setCard(false);
      env.render();
      return;
    }
    const ph = photos[selected];
    card.replaceChildren();
    for (const l of describePhoto(ph, input)) {
      card.append(Object.assign(document.createElement('p'), { textContent: l.v, className: l.k === 'title' ? 't' : l.k === 'range' || l.k === 'privacy' ? 's' : '' }));
    }
    // compare the photo with the model, or (no measured pose or field of view) show the photo alone and say why
    const [action] = cardActions(ph);
    const at = selected;
    const go = Object.assign(document.createElement('button'), {
      className: 'go', type: 'button', textContent: action.key === 'view' ? t('pp.viewFrom') : t('cmp.card.photo'),
    });
    go.addEventListener('click', () => viewFrom(at));
    if (action.key === 'photo') card.append(Object.assign(document.createElement('p'), { className: 's', textContent: noCompareText(action.note) }));
    const close = Object.assign(document.createElement('button'), { type: 'button', textContent: t('pp.close') });
    close.addEventListener('click', () => select(-1));
    card.append(go, close);
    cardButton = go;
    setCard(true);
    env.render();
  }

  // ---- fly / view from a photo
  let anim = null;
  function animateTo(toPos, toQuat, toFov, toNear, ms, done) {
    const fromPos = camera.position.clone();
    const fromQuat = camera.quaternion.clone();
    const fromFov = camera.fov;
    const t0 = performance.now();
    if (anim) cancelAnimationFrame(anim);
    const finish = () => {
      camera.near = toNear;
      camera.updateProjectionMatrix();
      camera.updateMatrixWorld();
      env.render();
    };
    if (ms <= 0) {
      camera.position.copy(toPos);
      camera.quaternion.copy(toQuat);
      camera.fov = toFov;
      finish();
      done?.();
      return;
    }
    const step = () => {
      const f = Math.min(1, (performance.now() - t0) / ms);
      const s = f < 0.5 ? 2 * f * f : 1 - (-2 * f + 2) ** 2 / 2;
      camera.position.lerpVectors(fromPos, toPos, s);
      camera.quaternion.slerpQuaternions(fromQuat, toQuat, s);
      camera.fov = fromFov + (toFov - fromFov) * s;
      finish();
      if (f < 1) anim = requestAnimationFrame(step);
      else {
        anim = null;
        done?.();
      }
    };
    anim = requestAnimationFrame(step);
  }

  // ---- "view from this photo" (web/compare.js): the photo and the model seen from its pose, side by side, stacked
  // or one over the other. The stage (renderer, labels) is resized to the model pane and the camera placed at the
  // photo's pose with its field of view at once, on entering and on every step, so photo and model change together.
  // A row without a measured pose or field of view shows its photo alone over the dimmed overview (photo-only): the
  // stage and the camera are left as they are.
  const stage = env.stage ?? renderer.domElement.parentElement;
  const prefs = comparePrefs(readComparePrefs(localStore()), opts);
  let cmp = null;
  let cardButton = null;
  let stageLabel = null;
  let barObserver = null;
  let laying = false;
  const STAGE_RECT = ['inset', 'left', 'top', 'right', 'bottom', 'width', 'height'];

  // the rows "view from this photo" accepts, by bearing from the centre: ‹ › step anticlockwise / clockwise (the
  // shown sources, and the current row)
  function ringOf(i) {
    return photos.map((p, k) => k).filter((k) => (visible[k] || k === i) && !viewFromBlocked(photos[k]))
      .sort((a, b) => Math.atan2(photos[a].p[0], photos[a].p[1]) - Math.atan2(photos[b].p[0], photos[b].p[1]));
  }

  function viewFrom(i) {
    if (typeof i === 'string') i = photos.findIndex((ph) => ph.id === i);
    if (!(i >= 0 && i < n)) return;
    const ph = photos[i];
    const photoOnly = viewFromBlocked(ph) !== null;
    if (viewing && viewing.photoOnly !== photoOnly) exitView(0);
    const entering = !viewing;
    if (entering && anim) {
      cancelAnimationFrame(anim);
      anim = null;
    }
    const back = viewing?.back ?? (photoOnly ? null
      : { pos: camera.position.clone(), quat: camera.quaternion.clone(), fov: camera.fov, near: camera.near, target: controls.target.clone() });
    viewing = { i, back, photoOnly, photo: 'loading', natural: null, placed: false };
    setCard(false);
    tip.hidden = true;
    hover = -1;
    dirty = true;
    if (entering) {
      // the language and light / dark switches (#prefs, in the page's panel) go into the comparison's bar meanwhile
      cmp = createCompareView(document, {
        mode: photoOnly ? 'photo' : prefs.mode, opacity: prefs.opacity, prefs: document.getElementById('prefs'),
        onStep: step, onExit: () => exitView(), onMode: setCompareMode, onOpacity: setCompareOpacity, onPhoto: photoEvent,
      });
      // the page's panels make way for the photo (under the photo-only scrim too, where they only got in the way)
      document.body.classList.add('pp-viewing');
      if (!photoOnly) {
        controls.enabled = false;
        document.body.classList.add('pp-compare');
        stageLabel = stage.getAttribute('aria-label');
        stage.setAttribute('aria-label', t('cmp.stage'));
      }
      window.addEventListener('resize', relayout);
      if (typeof ResizeObserver === 'function') {
        barObserver = new ResizeObserver(() => relayout());
        barObserver.observe(cmp.top);
        barObserver.observe(cmp.bar);
      }
    }
    const ring = photoOnly ? [] : ringOf(i);
    cmp.show({
      banner: compareBanner(ph, input.rounding), url: photoUrl(opts.url ?? '', ph.id), alt: t('cmp.alt', { src: SOURCE_LABEL[ph.src] }),
      k: ring.indexOf(i) + 1, n: ring.length,
    });
    relayout();
    if (entering) cmp.focus();
    syncUrl();
  }

  /**
   * The panes for the viewport and the bars as they are now: the stage to the model pane (the viewer's resize() at
   * once, so the renderer, the labels and the outline widths take its size before the next frame), the camera at the
   * row's pose with the pane's field of view, the photo on the model pane's photo frame (in the photo pane, or over
   * the model). On entering, every step, a mode change, a window resize or rotation and a change of the bars' size.
   */
  function relayout() {
    if (!viewing || !cmp) return;
    const ph = photos[viewing.i];
    const vw = window.innerWidth || document.documentElement.clientWidth;
    const vh = window.innerHeight || document.documentElement.clientHeight;
    const top = cmp.top.getBoundingClientRect().bottom;
    const bottom = vh - cmp.bar.getBoundingClientRect().top;
    if (viewing.photoOnly) {
      // the row's aspect, else the image's own once loaded, a 4:3 placeholder before
      const lay = compareLayout({ vw, vh, aspect: ph.aspect ?? viewing.natural ?? 4 / 3, mode: 'photo', top, bottom });
      cmp.place(lay, lay.photo);
      syncState();
      return;
    }
    const lay = compareLayout({ vw, vh, aspect: ph.aspect, mode: prefs.mode, top, bottom });
    const { x, y, w, h } = lay.model;
    laying = true;
    try {
      Object.assign(stage.style, { inset: 'auto', left: `${x}px`, top: `${y}px`, width: `${w}px`, height: `${h}px` });
      env.resize?.();
    } finally {
      laying = false;
    }
    const vp = viewPose(ph, w, h);
    camera.position.copy(pos[viewing.i]);
    camera.quaternion.copy(basisQuat[viewing.i]);
    camera.fov = vp.fov;
    camera.near = 0.3;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    const f = vp.frame;
    cmp.place(lay, { x: lay.photo.x + f.x * w, y: lay.photo.y + f.y * h, w: f.w * w, h: f.h * h });
    viewing.placed = true;
    env.render();
    syncState();
  }

  function step(dk) {
    if (!viewing || viewing.photoOnly) return;
    const ring = ringOf(viewing.i);
    if (ring.length < 2) return;
    viewFrom(ring[(ring.indexOf(viewing.i) + dk + ring.length) % ring.length]);
  }
  function setCompareMode(mode) {
    prefs.mode = mode;
    writeComparePrefs(localStore(), prefs);
    relayout();
    syncUrl();
  }
  function setCompareOpacity(opacity) {
    prefs.opacity = opacity;
    writeComparePrefs(localStore(), prefs);
    syncUrl();
  }
  function photoEvent(state, size) {
    if (!viewing) return;
    viewing.photo = state;
    const ph = photos[viewing.i];
    if (state === 'loaded' && size?.w > 0 && size?.h > 0) {
      const a = size.w / size.h;
      if (ph.aspect !== null && Math.abs(a - ph.aspect) > 0.02) {
        console.warn(`photo ${ph.id}: the image's aspect ${a.toFixed(3)} is not the row's ${ph.aspect}`);
      }
      if (viewing.photoOnly && ph.aspect === null) {
        viewing.natural = a;
        relayout();
      }
    }
    syncState();
  }
  // for the screenshot harness: what is shown, and whether the camera is placed and the photo loaded or failed
  function syncState() {
    env.state.compare = viewing ? { id: photos[viewing.i].id, mode: viewing.photoOnly ? 'photo' : prefs.mode, photo: viewing.photo } : null;
    env.state.posesViewReady = !!viewing && (viewing.photoOnly || viewing.placed) && viewing.photo !== 'loading';
  }
  // view=<id> and cmp (opacity in overlay mode) while a photo is shown, so a reload (a language switch) or a shared
  // link opens it again; removed on leaving
  function syncUrl() {
    try {
      const view = viewing ? { id: photos[viewing.i].id, mode: viewing.photoOnly ? 'photo' : prefs.mode, opacity: prefs.opacity } : null;
      const search = compareSearch(location.search, view);
      if (search !== location.search) history.replaceState(history.state, '', `${location.pathname}${search}${location.hash}`);
    } catch {
      /* no history here */
    }
  }

  /** Leaves the comparison (flies back over ms; at once with reduced motion) or closes a photo-only view. */
  function exitView(ms = reducedMotion() ? 0 : 600) {
    if (!viewing) return;
    const { i, back, photoOnly } = viewing;
    viewing = null;
    window.removeEventListener('resize', relayout);
    barObserver?.disconnect();
    barObserver = null;
    cmp?.destroy();
    cmp = null;
    dirty = true;
    document.body.classList.remove('pp-viewing');
    syncState();
    syncUrl();
    if (photoOnly) {
      select(i);
      cardButton?.focus();
      return;
    }
    document.body.classList.remove('pp-compare');
    for (const k of STAGE_RECT) stage.style.removeProperty(k);
    if (stageLabel === null) stage.removeAttribute('aria-label');
    else stage.setAttribute('aria-label', stageLabel);
    env.resize?.();
    animateTo(back.pos, back.quat, back.fov, back.near, ms, () => {
      controls.target.copy(back.target);
      controls.enabled = true;
      controls.update();
      select(i);
      cardButton?.focus();
    });
  }
  function flyToMembers(members) {
    const box = new THREE.Box3();
    for (const i of members) box.expandByPoint(pos[i]);
    const c = box.getCenter(new THREE.Vector3());
    const size = Math.max(6, box.getSize(new THREE.Vector3()).length());
    const dir = camera.position.clone().sub(controls.target).normalize();
    const target = c.clone().lerp(controls.target, 0.35);
    const toPos = target.clone().addScaledVector(dir, Math.max(size * 2.2, 12));
    const mm = new THREE.Matrix4().lookAt(toPos, target, new THREE.Vector3(0, 1, 0));
    const toQuat = new THREE.Quaternion().setFromRotationMatrix(mm);
    controls.enabled = false;
    animateTo(toPos, toQuat, camera.fov, camera.near, 600, () => {
      controls.target.copy(target);
      controls.enabled = true;
      controls.update();
      env.render();
    });
  }

  // the viewer's resize: the line widths' resolution; while comparing, the panes again (the window changed: the
  // stage's own size is the layout's)
  function resize() {
    for (const mtl of lineMats) mtl.resolution.copy(env.resolution);
    if (viewing && !laying) relayout();
  }

  /** Points the first view should include (three.js axes): the nearest 90 % of the rows, on the ground. */
  function fitPoints() {
    if (opts.fit === 'model' || !shown) return [];
    return fit.map((ph) => enuToThree(ph.p));
  }

  return {
    update,
    handleTap,
    resize,
    select,
    viewFrom,
    exitView,
    fitPoints,
    plazaR,
    get lastUpdateMs() {
      return lastUpdateMs;
    },
    setAllErrors(onoff) {
      allErrors = onoff;
      dirty = true;
    },
    setClustering(onoff) {
      clustering = onoff;
      dirty = true;
    },
    setShown(onoff) {
      shown = onoff;
      dirty = true;
    },
  };
}

function localStore() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

function reducedMotion() {
  try {
    return !!globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}

/**
 * Fetch and read a poses document (layerInput); null, with a console warning, when it is missing or not one. The
 * warnings are technical text for developers (the page shows no layer then), not translated.
 */
export async function loadPoses(url, fetchJson) {
  try {
    const input = layerInput(await fetchJson(url));
    if (!input) console.warn('photo poses: not a crowd3d.photo-poses version 2 document');
    return input;
  } catch (e) {
    console.warn(`photo poses could not be loaded: ${e.message ?? e}`);
    return null;
  }
}
