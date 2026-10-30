// Which model the landmark page lists first and shows in the 3D viewer, and the link to its stylized version.
// Pure, so node tests cover it.

export const STYLIZED_PENDING_TEXT = '风格化生成中';
export const STYLIZED_CLAY_LABEL = '素模（照片未对齐）'; // = crowd3d.api.STYLIZED_LABEL_CLAY

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
 * the server's label when its GLB is ready ({href, text, clay}); {pending: true, text} while a stylize job makes it;
 * null when there is none and none is coming. clay: the photos' registration was refused, so every wall is clay.
 */
export function stylizedView(model) {
  const s = model.stylized;
  if (!s) return null;
  if (s.status === 'pending') return { pending: true, text: STYLIZED_PENDING_TEXT };
  if (s.status !== 'ready' || !s.url) return null;
  const clay = !s.registration?.accepted;
  return {
    href: `stylized.html?model=${encodeURIComponent(s.url)}`,
    text: `风格化模型 · ${s.label || (clay ? STYLIZED_CLAY_LABEL : '照片已对齐')}`,
    clay,
  };
}

/** A key that changes whenever the model list must be drawn again (another model, or its stylized version). */
export function modelsKey(models) {
  return models.map((m) => `${m.id}:${m.stylized?.status ?? ''}:${m.stylized?.label ?? ''}`).join();
}
