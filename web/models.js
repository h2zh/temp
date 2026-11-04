// Which model the landmark page lists first and shows in the 3D viewer, and the link to its stylized version.
// Pure, so node tests cover it.

export const STYLIZED_PENDING_TEXT = '风格化生成中';
// The stylized model is always clay (D29); the label says whether the photos' registration was accepted.
export const STYLIZED_ALIGNED_LABEL = '素模（照片已对齐）'; // = crowd3d.api.STYLIZED_LABEL_ALIGNED
export const STYLIZED_UNALIGNED_LABEL = '素模（照片未对齐）'; // = crowd3d.api.STYLIZED_LABEL_UNALIGNED
// alignment_rmse_m: how far the cameras sit from their own GPS after the alignment, not the model's position error.
export const GPS_RESIDUAL_LABEL = 'GPS 对齐残差';

/** Full model first, then facades by sector; newest first within the same kind and sector. */
export function sortModels(models) {
  return [...models].sort(
    (a, b) =>
      (a.kind === 'full' ? 0 : 1) - (b.kind === 'full' ? 0 : 1) ||
      (a.sector ?? -1) - (b.sector ?? -1) ||
      b.created_at - a.created_at,
  );
}

/**
 * The model the viewer should switch to after a refresh, or null to keep what is on screen.
 * sorted: sortModels() output; shown: the model on screen or null; userPicked: the visitor chose it.
 * First visit: the best model. Later: the rebuilt version of the model on screen, and the full
 * model as soon as it exists unless the visitor picked a model themselves.
 */
export function modelToShow(sorted, shown, userPicked) {
  if (!sorted.length) return null;
  if (!shown) return sorted[0];
  if (!userPicked && sorted[0].kind === 'full' && shown.kind !== 'full') return sorted[0];
  const same = sorted.find((m) => m.kind === shown.kind && m.sector === shown.sector);
  return same && same.id !== shown.id ? same : null;
}

/**
 * What the model list says about a model's stylized version (ModelInfo.stylized): a link to the stylized viewer with
 * the server's label when its GLB is ready ({href, text, unaligned}; the link carries the landmark's public photo poses,
 * poses_url, as &poses=); {pending: true, text} while a stylize job
 * makes it; null when there is none and none is coming. The model is clay either way (D29). unaligned: the photos'
 * registration was refused, so D15 and D24 did not use the photos (D25 corrections stay by their grade).
 */
export function stylizedView(model) {
  const s = model.stylized;
  if (!s) return null;
  if (s.status === 'pending') return { pending: true, text: STYLIZED_PENDING_TEXT };
  if (s.status !== 'ready' || !s.url) return null;
  const unaligned = !s.registration?.accepted;
  // the public photo poses in the GLB's frame (D30): a same-origin path only
  const poses = typeof s.poses_url === 'string' && s.poses_url.startsWith('/api/') ? s.poses_url : null;
  return {
    href: `stylized.html?model=${encodeURIComponent(s.url)}${poses ? `&poses=${encodeURIComponent(poses)}` : ''}`,
    text: `风格化模型 · ${s.label || (unaligned ? STYLIZED_UNALIGNED_LABEL : STYLIZED_ALIGNED_LABEL)}`,
    unaligned,
  };
}

/** A key that changes whenever the model list must be drawn again (another model, or its stylized version). */
export function modelsKey(models) {
  return models.map((m) => `${m.id}:${m.stylized?.status ?? ''}:${m.stylized?.label ?? ''}`).join();
}

/**
 * The model's horizontal position 1-sigma in metres at its points (ModelInfo.geo.position_sigma_m.horizontal), or null
 * when it is not to be shown: only for a fully georegistered model whose anchor calibration passed the held-out check
 * (geo.calibrated, from the server's Settings.calibrated_anchor_ids, empty until the final evaluation passes one).
 */
export function positionSigmaM(model) {
  const g = model.geo;
  if (!g || g.calibrated !== true || g.level !== 'full') return null;
  const h = g.position_sigma_m?.horizontal;
  return typeof h === 'number' && Number.isFinite(h) ? h : null;
}
