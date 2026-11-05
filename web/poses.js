// The public photo poses of a landmark (GET /api/landmarks/{id}/poses, format crowd3d.photo-poses version 2, D30):
// parsing and the texts the landmark page and the clay viewer's photo layer (photoposes.js) show, in the page language
// (web/strings.js poses.*). Pure, so node tests cover it.
//
// Only photos whose uploaders agreed are in it, rounded by the server (whole metres, whole degrees) and keyed by a
// salted public id; nothing here rounds again or guesses a missing value.

import { SECTOR_LABELS, sectorShort } from './compass.js';
import { fmtPercent, keyed, t } from './i18n.js';

export const POSES_FORMAT = 'crowd3d.photo-poses';
export const POSES_VERSION = 2;
export const SOURCES = ['sfm', 'loc', 'exif'];

// What each source means (crowd3d/photoposes.py); the EXIF label is the owner's wording (D30 question 1).
export const SOURCE_TEXT = keyed('poses.source', SOURCES);

export const SOURCE_HINT = keyed('poses.sourceHint', SOURCES);

const isInt = (v) => Number.isInteger(v);

/**
 * The rows of a poses document, or null when it is not one this page understands. A row with an unknown source, or
 * without a whole-metre position, a heading and its 2-sigma ranges (whole sp metres, whole sy degrees), is dropped:
 * never drawn from a guess, and never without the range that says how far off it may be.
 */
export function parsePoses(doc) {
  if (!doc || doc.format !== POSES_FORMAT || doc.version !== POSES_VERSION || !Array.isArray(doc.photos)) return null;
  const rows = doc.photos.filter(
    (r) =>
      r &&
      typeof r.id === 'string' &&
      SOURCES.includes(r.src) &&
      Array.isArray(r.p) &&
      r.p.length === 3 &&
      isInt(r.p[0]) &&
      isInt(r.p[1]) &&
      (r.p[2] === null || isInt(r.p[2])) &&
      isInt(r.yaw) &&
      isInt(r.sp) &&
      r.sp >= 0 &&
      isInt(r.sy) &&
      r.sy >= 0,
  );
  return { ...doc, photos: rows };
}

/** Markers for the point-cloud viewer (viewer.setCameras): an EXIF row has no measured height, so it sits on the ground. */
export function poseMarkers(rows) {
  return rows.map((r) => ({ enu: [r.p[0], r.p[1], r.p[2] ?? 0], heading_deg: r.yaw }));
}

/** What the landmark page says instead of the markers on a snapped model's point cloud (cloudMarkers). */
export function snappedCloudNote() {
  return t('poses.snappedNote');
}

/**
 * The markers for the point cloud of the model on screen: none when that model's stylized GLB was snapped to the
 * footprint (ModelInfo.stylized.snapped). The public poses moved that model's cameras with the snap, into the GLB's
 * frame, but its point cloud did not move, so they would sit off the cloud by the snap; the stylized model shows them.
 */
export function cloudMarkers(rows, model) {
  return model?.stylized?.snapped === true ? [] : poseMarkers(rows);
}

/** A key that changes when the markers must be drawn again. */
export function markersKey(rows) {
  return rows.map((r) => `${r.id}:${r.p.join(',')}:${r.yaw}`).join(';');
}

/** "东 12 米，南 3 米" / "12 m east, 3 m south" from the landmark centre. */
export function positionText(p) {
  const [e, n] = p;
  const params = { ew: Math.abs(e), ns: Math.abs(n) };
  if (e < 0) return n < 0 ? t('poses.position.ws', params) : t('poses.position.wn', params);
  return n < 0 ? t('poses.position.es', params) : t('poses.position.en', params);
}

/** "183°（南）" / "183° (S)": the compass bearing the camera looked along. */
export function headingText(yaw) {
  return t('poses.heading', { deg: String(yaw), dir: sectorShort(SECTOR_LABELS[Math.round(yaw / 45) % 8]) });
}

/** "±16 米 · ±31°" / "±16 m · ±31°": the row's 2-sigma ranges. */
export function rangeText(row) {
  const parts = [];
  if (isInt(row.sp)) parts.push(t('poses.range.position', { n: row.sp }));
  if (isInt(row.sy)) parts.push(`±${row.sy}°`);
  return parts.join(' · ') || '—';
}

/**
 * The count line: how many of the snapshot's accepted photos are public (uploaded under the user agreement and not
 * withdrawn: counts.shared of counts.photos), how many of them have a pose shown, why the rest are not.
 */
export function countsText(counts) {
  const c = counts || {};
  const params = { shared: c.shared ?? 0, n: c.photos ?? 0, shown: c.shown ?? 0, undetermined: c.not_determined ?? 0 };
  if (c.shared) return c.not_determined ? t('poses.counts.shownUndetermined', params) : t('poses.counts.shown', params);
  return c.not_determined ? t('poses.counts.undetermined', params) : t('poses.counts', params);
}

/** A measured share {within, n} the page can quote: whole counts, 0 <= within <= n, n > 0. */
function isShare(m) {
  return !!m && isInt(m.n) && m.n > 0 && isInt(m.within) && m.within >= 0 && m.within <= m.n;
}

/** "165 个中 146 个（88%）" / "146 of 165 (88%)" */
function shareText(m) {
  return t('poses.share', { n: m.n, within: m.within, percent: fmtPercent(Math.round((100 * m.within) / m.n) / 100) });
}

/**
 * How often the ranges held, per source, from the server's measurement (uncertainty.measured_coverage): only the
 * sources with rows here, "还没有实测" where nothing was measured. Per source: the pooled share, the worst and best
 * single landmark-run (lowest / highest), and whether the source's range rule was calibrated on those same landmarks
 * (rule_calibrated_here: then the share is no independent check). Nothing is claimed that the server did not send.
 * labels: the sources' names (the clay viewer's photo layer passes its own, photoposes.js SOURCE_LABEL).
 */
export function coverageText(uncertainty, rows, labels = SOURCE_TEXT) {
  const present = SOURCES.filter((s) => rows.some((r) => r.src === s));
  if (!present.length) return '';
  const sigmas = uncertainty?.sigmas ?? 2;
  const parts = present.map((s) => t('poses.coverage.source', { source: labels[s], coverage: sourceCoverageText(uncertainty, s) }));
  return t('poses.coverage', { sigmas, sources: parts.join(t('common.sentenceSep')) });
}

/**
 * One source's measured share as coverageText says it, without the source's name and the closing full stop (the
 * clay viewer's card quotes it for one photo): "4 个测试地标、2 次运行合计 165 个中 146 个（88%），单个地标单次运行最低
 * …；这个来源的范围规则没有用这些地标调过", or "还没有实测" ("146 of 165 (88%) on 4 test landmarks over 2 runs; …",
 * "not measured yet").
 */
export function sourceCoverageText(uncertainty, src) {
  const m = uncertainty?.measured_coverage?.[src];
  if (!isShare(m)) return t('poses.measured.none');
  const share = shareText(m);
  const landmarks = isInt(m.landmarks) && m.landmarks > 0 ? m.landmarks : null;
  const runs = isInt(m.runs) && m.runs > 1 ? m.runs : null;
  let text;
  if (landmarks !== null) {
    text = runs !== null
      ? t('poses.measured.landmarksRuns', { n: landmarks, runs, share })
      : t('poses.measured.landmarks', { n: landmarks, share });
  } else {
    text = runs !== null ? t('poses.measured.runs', { runs, share }) : t('poses.measured', { share });
  }
  if (isShare(m.lowest) && isShare(m.highest)) {
    text = t('poses.measured.spread', { text, lowest: shareText(m.lowest), highest: shareText(m.highest) });
  }
  if (m.rule_calibrated_here === true) text = t('poses.measured.calibratedHere', { text });
  else if (m.rule_calibrated_here === false) text = t('poses.measured.notCalibratedHere', { text });
  return text;
}

/** Which sources the rows have, explained. */
export function sourcesText(rows) {
  const present = SOURCES.filter((s) => rows.some((r) => r.src === s));
  return present.map((s) => t('poses.sourceLine', { source: SOURCE_TEXT[s], hint: SOURCE_HINT[s] })).join(t('common.sentenceSep'));
}
