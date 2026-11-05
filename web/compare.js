// "View from this photo" on the clay model (D32): the uploaded photo and the model seen from the photo's pose, side
// by side (stacked on a narrow screen) or one over the other with an opacity slider, so their alignment can be
// compared. web/photoposes.js opens it from a photo's card and moves the 3-D view: it resizes the model's stage to the
// model pane and places the camera there with the photo's field of view; this module lays the panes out, draws the
// bars, the photo and its states, and says what is shown. No three.js here: everything is node-testable.
//
// The photo comes from GET /api/landmarks/{id}/poses/{public_id}/photo (photoUrl: the poses document's own path plus
// the row's 12-hex public id, never a store id), served metadata-free and downscaled for the rows of the current
// poses snapshot only; any other request, a withdrawn photo included, gets a 404, shown as cmp.missing while the
// model view stays. One photo is fetched when it is shown, no neighbour ahead of time.
//
// D30 holds: only a row with a measured position, height, pitch, roll, field of view and aspect is compared with the
// model (viewFromBlocked); any other row (EXIF, or a reconstructed or matched row missing one of them) opens its photo
// alone with its source, its 2-sigma ranges and why it cannot be compared (photo-only mode), never a model view from
// an assumed value. Nothing about who took the photo or when is shown.
//
// Colours: the bars, chips and panes read stylized.html's --panel-* tokens and follow light / dark; behind the model
// pane is the art theme's sky (stylized.html), whatever the mode.
//
// This module and photoposes.js import each other; neither uses the other's exports at module top level, only inside
// functions, so either may be loaded first.

import { fmtPercent, t } from './i18n.js';
import { BLOCKED_TEXT, DISC_SP_M, SOURCE_LABEL, parsePoseParams, roundingText, viewFromBlocked } from './photoposes.js';

/** A row's public id: 12 lower-case hex digits (crowd3d/photoposes.py; a store id is 32 and never reaches the page). */
export const PUBLIC_ID_RE = /^[0-9a-f]{12}$/;
/** The two comparison modes the bar switches between (a blocked row's photo-only mode is 'photo'). */
export const MODES = ['pair', 'overlay'];
export const DEFAULT_MODE = 'pair';
export const DEFAULT_OPACITY = 50; // the photo's opacity in overlay mode, percent
export const OPACITY_STEP = 5;
/** localStorage key of the chosen mode and opacity ({mode, opacity}); URL parameters cmp and opacity override it. */
export const PREFS_KEY = 'crowd3d.compare';

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

/**
 * The URL of a row's photo: the poses document's path without its query, then /<public id>/photo; null for an id that
 * is not a public one or a poses path parsePoseParams would not accept (another origin, not a URL).
 */
export function photoUrl(posesPath, id) {
  if (typeof id !== 'string' || !PUBLIC_ID_RE.test(id) || typeof posesPath !== 'string' || !posesPath) return null;
  const url = parsePoseParams(`?${new URLSearchParams({ poses: posesPath })}`).url;
  if (!url) return null;
  const path = url.split(/[?#]/)[0].replace(/\/+$/, '');
  return path ? `${path}/${id}/photo` : null;
}

/** An opacity from the URL or storage: 0..100 in steps of 5 (clamped), or null when it is no number. */
export function parseOpacity(v) {
  let x = NaN;
  if (finite(v)) x = v;
  else if (typeof v === 'string' && /^\s*-?\d+(\.\d+)?\s*$/.test(v)) x = Number(v);
  if (!Number.isFinite(x)) return null;
  return Math.max(0, Math.min(100, Math.round(x / OPACITY_STEP) * OPACITY_STEP));
}

/** The comparison's URL parameters (URLSearchParams): cmp=pair|overlay, opacity=0..100; null where absent or invalid. */
export function parseCompareParams(params) {
  const mode = params.get('cmp');
  return { cmp: MODES.includes(mode) ? mode : null, opacity: parseOpacity(params.get('opacity')) };
}

/** The stored choice ({mode, opacity}, each null when absent or invalid); a blocked or broken storage gives nulls. */
export function readComparePrefs(storage) {
  try {
    const raw = storage?.getItem(PREFS_KEY);
    const v = raw ? JSON.parse(raw) : null;
    return { mode: MODES.includes(v?.mode) ? v.mode : null, opacity: parseOpacity(v?.opacity) };
  } catch {
    return { mode: null, opacity: null };
  }
}

/** Keeps the choice; a blocked storage keeps it for this page only (the caller's copy). */
export function writeComparePrefs(storage, { mode, opacity }) {
  try {
    storage?.setItem(PREFS_KEY, JSON.stringify({ mode, opacity }));
  } catch {
    /* private mode or storage blocked */
  }
}

/** The mode and opacity to start with: the URL's (parsePoseParams cmp, opacity), else the stored ones, else the defaults. */
export function comparePrefs(stored, params = {}) {
  return {
    mode: params.cmp ?? stored?.mode ?? DEFAULT_MODE,
    opacity: params.opacity ?? stored?.opacity ?? DEFAULT_OPACITY,
  };
}

/**
 * location.search while comparing ({id, mode, opacity}: view=<id>, cmp=<mode>, and opacity=<n> in overlay mode; a
 * photo-only row only view=<id>) or after it (null: those removed). The other parameters keep their text.
 */
export function compareSearch(search, view) {
  const kept = String(search ?? '')
    .replace(/^\?/, '')
    .split('&')
    .filter((kv) => kv && !/^(view|cmp|opacity)(=|$)/.test(kv));
  if (view && PUBLIC_ID_RE.test(view.id ?? '')) {
    kept.push(`view=${view.id}`);
    if (MODES.includes(view.mode)) kept.push(`cmp=${view.mode}`);
    if (view.mode === 'overlay' && parseOpacity(view.opacity) !== null) kept.push(`opacity=${parseOpacity(view.opacity)}`);
  }
  return kept.length ? `?${kept.join('&')}` : '';
}

/**
 * Where the panes go in a viewport of vw x vh CSS pixels with the top bar's bottom edge at `top` and the bottom bar
 * `bottom` pixels high, for a photo of `aspect` (width / height): {dir, photo, model}, rectangles {x, y, w, h} in whole
 * pixels. 'pair': the photo and the model pane, the same size and the photo's aspect, side by side (dir 'row') or
 * stacked ('column'), whichever gives the larger panes (a tie: side by side); the photo left or on top, the group
 * centred. 'overlay' and 'photo': one pane, photo === model. dir is the pair's direction in every mode (the mode
 * button's label).
 */
export function compareLayout({ vw, vh, aspect, mode, top = 0, bottom = 0, pad = 12, gap = 10 }) {
  const a = finite(aspect) && aspect > 0 ? aspect : 4 / 3;
  const T = Math.ceil(top);
  const W = Math.max(1, Math.floor(vw - 2 * pad));
  const H = Math.max(1, Math.floor(vh - T - Math.ceil(bottom) - 2 * pad));
  // the tallest pane of the photo's aspect that fits, its width rounded up: w / h is the aspect or up to 1/h wider, so
  // the photo's frame in the pane spans its full height and all but under 1 px of its width (photoposes.js
  // photoFrame), and the camera's vertical field of view is the photo's own
  const fit = (aw, ah) => {
    const h = Math.max(1, Math.min(ah, Math.floor(aw / a)));
    return { w: Math.max(1, Math.min(aw, Math.ceil(a * h - 1e-9))), h };
  };
  const side = fit(Math.max(1, Math.floor((W - gap) / 2)), H);
  const stack = fit(W, Math.max(1, Math.floor((H - gap) / 2)));
  const dir = stack.w * stack.h > side.w * side.h ? 'column' : 'row';
  const x0 = (gw) => pad + Math.floor((W - gw) / 2);
  const y0 = (gh) => T + pad + Math.floor((H - gh) / 2);
  if (mode === 'pair') {
    const { w, h } = dir === 'row' ? side : stack;
    const x = x0(dir === 'row' ? 2 * w + gap : w);
    const y = y0(dir === 'row' ? h : 2 * h + gap);
    const photo = { x, y, w, h };
    const model = dir === 'row' ? { x: x + w + gap, y, w, h } : { x, y: y + h + gap, w, h };
    return { dir, photo, model };
  }
  const one = fit(W, H);
  const rect = { x: x0(one.w), y: y0(one.h), w: one.w, h: one.h };
  return { dir, photo: rect, model: rect };
}

/**
 * What the card offers for a row: [{key: 'view'}] (compare the photo with the model) for a row with a measured pose
 * and field of view, else [{key: 'photo', note: <viewFromBlocked code>}] (its photo alone; noCompareText(note) says
 * why).
 */
export function cardActions(ph) {
  const code = viewFromBlocked(ph);
  return code ? [{ key: 'photo', note: code }] : [{ key: 'view' }];
}

/** The card's line under a photo-only button: why the row cannot be compared with the model. */
export function noCompareText(code) {
  return t('cmp.card.noCompare', { reason: BLOCKED_TEXT[code] });
}

const sentenceCase = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

/**
 * The top bar for a row: {title, detail, warn}. Compared rows: the source, the 2-sigma ranges, the rounding (with the
 * field-of-view step) and that the clay model has no detail; warn (else null) when the position range is wider than
 * DISC_SP_M. A photo-only row: its source and ranges and why it cannot be compared. Only the row's public values.
 */
export function compareBanner(ph, rounding = null) {
  const code = viewFromBlocked(ph);
  const base = { src: SOURCE_LABEL[ph.src], sp: ph.sp, sy: ph.sy };
  if (code) {
    return { title: t('cmp.photoOnly.title'), detail: t('cmp.photoOnly.detail', { ...base, reason: sentenceCase(BLOCKED_TEXT[code]) }), warn: null };
  }
  return {
    title: t('cmp.title'),
    detail: t('cmp.detail', { ...base, rounding: roundingText(rounding) }),
    warn: ph.sp > DISC_SP_M ? t('cmp.warn') : null,
  };
}

// ------------------------------------------------------------------ DOM

// Only the panel tokens (stylized.html, both modes) and the photo-only scrim; the model pane's sky is stylized.html's.
const CSS = `
body.pp-compare{background:var(--panel-page)}
.pp-cmp{position:fixed;inset:0;z-index:6;pointer-events:none;font-size:13px;color:var(--panel-fg)}
.pp-cmp[data-mode="photo"]{background:rgba(0,0,0,.55);pointer-events:auto}
.pp-cmp [hidden]{display:none!important}
.pp-cmp-top,.pp-cmp-bar{position:absolute;z-index:4;left:50%;transform:translateX(-50%);box-sizing:border-box;width:max-content;
  max-width:min(760px,calc(100vw - 16px));pointer-events:auto;padding:6px 12px;border-radius:16px;background:var(--panel);
  color:var(--panel-fg);border:3px solid var(--panel-line);box-shadow:0 4px 0 var(--panel-line);
  -webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px)}
.pp-cmp-top{top:max(8px,env(safe-area-inset-top));text-align:center}
.pp-cmp-top p{margin:0}
.pp-cmp-title{font-weight:800;font-size:14px}
.pp-cmp-detail{font-size:12px;margin-top:2px!important}
.pp-cmp-warn{font-size:12px;font-weight:700;color:var(--panel-note);margin-top:2px!important}
.pp-cmp-bar{bottom:max(12px,env(safe-area-inset-bottom));display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:6px 12px}
.pp-cmp-bar button{font:inherit;font-weight:800;padding:4px 10px;border-radius:999px;border:2.5px solid var(--panel-line);
  background:var(--panel-chip);color:var(--panel-fg);cursor:pointer;white-space:nowrap}
.pp-cmp-bar button[aria-pressed="true"]{background:var(--panel-go)}
.pp-cmp-bar button:disabled{opacity:.45;cursor:not-allowed}
.pp-cmp-nav,.pp-cmp-modes,.pp-cmp-op{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.pp-cmp-modes{gap:4px}
.pp-cmp-count{font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}
.pp-cmp-op{font-weight:700}
.pp-cmp-op input{width:120px;margin:0;accent-color:var(--panel-fg)}
.pp-cmp-bar #prefs{margin:0}
.pp-cmp-pane,.pp-cmp-model{position:absolute;box-sizing:border-box;outline:3px solid var(--panel-line);pointer-events:none}
.pp-cmp-pane{background:var(--panel-chip)}
.pp-cmp[data-mode="overlay"] .pp-cmp-pane,.pp-cmp[data-photo="loaded"] .pp-cmp-pane{background:transparent}
.pp-cmp[data-mode="overlay"] .pp-cmp-model,.pp-cmp[data-mode="photo"] .pp-cmp-model,.pp-cmp[data-mode="overlay"] .pp-cmp-chip{display:none}
.pp-cmp-photo{position:absolute;z-index:2;display:block;pointer-events:none;object-fit:contain}
.pp-cmp[data-photo="loading"] .pp-cmp-photo,.pp-cmp[data-photo="missing"] .pp-cmp-photo{visibility:hidden}
.pp-cmp-chip{position:absolute;left:8px;top:8px;z-index:3;padding:1px 8px;border-radius:999px;background:var(--panel-chip);
  color:var(--panel-fg);border:2px solid var(--panel-line);font-weight:800;font-size:12px;white-space:nowrap}
.pp-cmp-state{position:absolute;left:50%;top:50%;z-index:3;transform:translate(-50%,-50%);box-sizing:border-box;margin:0;
  width:max-content;max-width:calc(100% - 24px);padding:6px 10px;border-radius:12px;background:var(--panel);color:var(--panel-fg);
  border:2px solid var(--panel-line);font-weight:700;text-align:center}
.pp-cmp[data-photo="loaded"] .pp-cmp-state{display:none}
@media (pointer:coarse){.pp-cmp-bar button{min-height:44px;min-width:44px}.pp-cmp-op input{height:44px}.pp-cmp-bar select{min-height:44px}}
`;

const styled = new WeakSet();

const placeAt = (node, r) => Object.assign(node.style, { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` });

/**
 * The comparison's DOM in doc: <div class="pp-cmp" data-mode="pair|overlay|photo" data-photo="loading|loaded|missing"
 * role="region">: the top bar (aria-live), the photo pane with its chip and state line, the model pane's frame and
 * chip (the model itself is the page's stage, under it), the photo and the bottom bar (prev / count / next, the mode
 * buttons, the opacity slider in overlay mode, Back; a photo-only view has only Close), then the page's display
 * settings: prefs, the language and light / dark switches (#prefs, web/ui.js mountPrefs), is moved to the end of the
 * bar while the view is open and put back where it was on destroy, so both switches stay in reach over the photo (the
 * page's panels are hidden or under the photo-only scrim then; D32: the switches on every page). Escape calls onExit,
 * ArrowLeft / ArrowRight onStep(-1 / +1) (not from a form field, not in photo-only mode). Returns {el, top, bar, img,
 * show(row), setMode(mode), setOpacity(percent), setPhotoState(state), place(layout, imgRect), focus(), destroy()};
 * row: {banner: compareBanner(), url: photoUrl(), alt, k, n} (k of n: the row's place on the ring around the landmark).
 */
export function createCompareView(doc, { mode = DEFAULT_MODE, opacity = DEFAULT_OPACITY, prefs = null, onStep, onExit, onMode, onOpacity, onPhoto } = {}) {
  if (!styled.has(doc)) {
    const style = doc.createElement('style');
    style.textContent = CSS;
    doc.head.append(style);
    styled.add(doc);
  }
  const make = (tag, props = {}, attrs = {}) => {
    const node = Object.assign(doc.createElement(tag), props);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  };
  const photoOnly = mode === 'photo';
  let current = photoOnly ? 'photo' : MODES.includes(mode) ? mode : DEFAULT_MODE;
  let level = parseOpacity(opacity) ?? DEFAULT_OPACITY;
  let img = null;
  let imgRect = null;

  const root = make('div', { className: 'pp-cmp' }, { role: 'region', 'aria-label': t('cmp.region') });
  const top = make('div', { className: 'pp-cmp-top' }, { 'aria-live': 'polite' });
  const title = make('p', { className: 'pp-cmp-title' });
  const detail = make('p', { className: 'pp-cmp-detail' });
  const warn = make('p', { className: 'pp-cmp-warn', hidden: true });
  top.append(title, detail, warn);

  const pane = make('div', { className: 'pp-cmp-pane' });
  const state = make('p', { className: 'pp-cmp-state' });
  pane.append(make('span', { className: 'pp-cmp-chip', textContent: t('cmp.chip.photo') }), state);
  const modelBox = make('div', { className: 'pp-cmp-model' });
  modelBox.append(make('span', { className: 'pp-cmp-chip', textContent: t('cmp.chip.model') }));

  const bar = make('div', { className: 'pp-cmp-bar' });
  const button = (text, cls, onClick, attrs = {}) => {
    const b = make('button', { textContent: text, className: cls }, { type: 'button', ...attrs });
    b.addEventListener('click', onClick);
    return b;
  };
  const prev = button('‹', 'pp-cmp-prev', () => onStep?.(-1), { title: t('cmp.prev'), 'aria-label': t('cmp.prev') });
  const next = button('›', 'pp-cmp-next', () => onStep?.(1), { title: t('cmp.next'), 'aria-label': t('cmp.next') });
  const count = make('span', { className: 'pp-cmp-count' });
  const nav = make('span', { className: 'pp-cmp-nav' });
  nav.append(prev, count, next);
  const choose = (m) => {
    if (m === current) return;
    setMode(m);
    onMode?.(m);
  };
  const pairBtn = button(t('cmp.pair.row'), 'pp-cmp-mode', () => choose('pair'));
  const overlayBtn = button(t('cmp.overlay'), 'pp-cmp-mode', () => choose('overlay'));
  pairBtn.dataset.mode = 'pair';
  overlayBtn.dataset.mode = 'overlay';
  const modes = make('span', { className: 'pp-cmp-modes' });
  modes.append(pairBtn, overlayBtn);
  const range = make('input', {}, { type: 'range', min: '0', max: '100', step: String(OPACITY_STEP), 'aria-label': t('cmp.opacity') });
  range.addEventListener('input', () => {
    const v = parseOpacity(Number(range.value));
    if (v === null) return;
    setOpacity(v);
    onOpacity?.(v);
  });
  const op = make('span', { className: 'pp-cmp-op' });
  op.append(make('span', { textContent: t('cmp.opacity.model') }, { 'aria-hidden': 'true' }), range,
    make('span', { textContent: t('cmp.opacity.photo') }, { 'aria-hidden': 'true' }));
  const back = button(photoOnly ? t('pp.close') : t('cmp.back'), 'pp-cmp-back', () => onExit?.());
  if (photoOnly) {
    nav.hidden = true;
    modes.hidden = true;
  }
  bar.append(nav, modes, op, back);
  // the display settings: borrowed from their place on the page while the view is open
  const prefsHome = prefs?.parentNode ? { parent: prefs.parentNode, next: prefs.nextSibling ?? null } : null;
  if (prefs) bar.append(prefs);
  root.append(top, pane, modelBox, bar);

  function applyOpacity() {
    if (img) img.style.opacity = current === 'overlay' ? String(level / 100) : '1';
  }
  function setPhotoState(s) {
    root.dataset.photo = s;
    state.textContent = s === 'loading' ? t('cmp.loading') : s === 'missing' ? t('cmp.missing') : '';
    state.hidden = s === 'loaded';
    range.disabled = s !== 'loaded';
  }
  function setMode(m) {
    if (photoOnly || !MODES.includes(m)) return;
    current = m;
    root.dataset.mode = m;
    pairBtn.setAttribute('aria-pressed', String(m === 'pair'));
    overlayBtn.setAttribute('aria-pressed', String(m === 'overlay'));
    op.hidden = m !== 'overlay';
    applyOpacity();
  }
  function setOpacity(v) {
    const x = parseOpacity(v);
    if (x === null) return;
    level = x;
    range.value = String(x);
    range.setAttribute('aria-valuetext', fmtPercent(x / 100));
    applyOpacity();
  }
  root.dataset.mode = current;
  op.hidden = current !== 'overlay';
  if (!photoOnly) setMode(current);
  setOpacity(level);
  setPhotoState('loading');

  function show({ banner, url, alt = '', k = 0, n = 0 }) {
    title.textContent = banner.title;
    detail.textContent = banner.detail;
    warn.textContent = banner.warn ?? '';
    warn.hidden = !banner.warn;
    count.textContent = k > 0 && n > 0 ? t('cmp.count', { k, n }) : '';
    prev.disabled = next.disabled = !(n > 1);
    // a new image element per row: an old one's late load or error never reaches the new row
    img?.remove();
    const el = make('img', { className: 'pp-cmp-photo', alt, decoding: 'async', draggable: false });
    img = el;
    el.addEventListener('load', () => {
      if (img !== el) return;
      setPhotoState('loaded');
      onPhoto?.('loaded', { w: el.naturalWidth, h: el.naturalHeight });
    });
    el.addEventListener('error', () => {
      if (img !== el) return;
      setPhotoState('missing');
      onPhoto?.('missing', null);
    });
    setPhotoState('loading');
    applyOpacity();
    if (imgRect) placeAt(el, imgRect);
    root.append(el);
    if (url) el.src = url;
    else {
      setPhotoState('missing');
      onPhoto?.('missing', null);
    }
  }

  /** layout: compareLayout(); rect: where the photo goes (the model pane's photo frame, in the photo pane). */
  function place(layout, rect) {
    placeAt(pane, layout.photo);
    placeAt(modelBox, layout.model);
    imgRect = rect;
    if (img) placeAt(img, rect);
    pairBtn.textContent = layout.dir === 'column' ? t('cmp.pair.column') : t('cmp.pair.row');
  }

  const FIELDS = /^(INPUT|SELECT|TEXTAREA)$/;
  function onKey(e) {
    if (e.key === 'Escape' || e.key === 'Esc') {
      e.preventDefault?.();
      onExit?.();
      return;
    }
    if (photoOnly || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
    if (e.altKey || e.ctrlKey || e.metaKey || FIELDS.test(e.target?.tagName ?? '') || e.target?.isContentEditable) return;
    e.preventDefault?.();
    onStep?.(e.key === 'ArrowRight' ? 1 : -1);
  }
  doc.addEventListener('keydown', onKey);
  doc.body.append(root);

  return {
    el: root,
    top,
    bar,
    get img() {
      return img;
    },
    show,
    setMode,
    setOpacity,
    setPhotoState,
    place,
    focus() {
      back.focus();
    },
    destroy() {
      doc.removeEventListener('keydown', onKey);
      if (prefs && prefs.parentNode === bar) {
        if (prefsHome) prefsHome.parent.insertBefore(prefs, prefsHome.next?.parentNode === prefsHome.parent ? prefsHome.next : null);
        else prefs.remove();
      }
      root.remove();
    },
  };
}
