// Which model the landmark page lists first and shows in the 3D viewer. Pure, so node tests cover it.

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
