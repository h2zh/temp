// The public photo poses of a landmark (GET /api/landmarks/{id}/poses, format crowd3d.photo-poses version 2, D30):
// parsing and the Chinese texts the landmark page shows. Pure, so node tests cover it.
//
// Only photos whose uploaders agreed are in it, rounded by the server (whole metres, whole degrees) and keyed by a
// salted public id; nothing here rounds again or guesses a missing value.

export const POSES_FORMAT = 'crowd3d.photo-poses';
export const POSES_VERSION = 2;
export const SOURCES = ['sfm', 'loc', 'exif'];

// What each source means (crowd3d/photoposes.py); the EXIF label is the owner's wording (D30 question 1).
export const SOURCE_TEXT = {
  sfm: '模型重建',
  loc: '与模型比对',
  exif: '手机 GPS + 罗盘（未经模型核对）',
};

export const SOURCE_HINT = {
  sfm: '照片参与了三维重建，位置和朝向由重建算出',
  loc: '照片与已建好的模型比对算出',
  exif: '照片自带的定位和指南针，没有和模型核对，范围大得多',
};

const DIRECTIONS = ['北', '东北', '东', '东南', '南', '西南', '西', '西北'];

const isInt = (v) => Number.isInteger(v);

/**
 * The rows of a poses document, or null when it is not one this page understands. A row with an unknown source or
 * without a whole-metre position and a heading is dropped (never drawn from a guess).
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
      isInt(r.yaw),
  );
  return { ...doc, photos: rows };
}

/** Markers for the point-cloud viewer (viewer.setCameras): an EXIF row has no measured height, so it sits on the ground. */
export function poseMarkers(rows) {
  return rows.map((r) => ({ enu: [r.p[0], r.p[1], r.p[2] ?? 0], heading_deg: r.yaw }));
}

/** A key that changes when the markers must be drawn again. */
export function markersKey(rows) {
  return rows.map((r) => `${r.id}:${r.p.join(',')}:${r.yaw}`).join(';');
}

/** "东 12 米，南 3 米" from the landmark centre. */
export function positionText(p) {
  const [e, n] = p;
  return `${e < 0 ? '西' : '东'} ${Math.abs(e)} 米，${n < 0 ? '南' : '北'} ${Math.abs(n)} 米`;
}

/** "183°（南）": the compass bearing the camera looked along. */
export function headingText(yaw) {
  return `${yaw}°（${DIRECTIONS[Math.round(yaw / 45) % 8]}）`;
}

/** "±16 米 · ±31°": the row's 2-sigma ranges. */
export function rangeText(row) {
  const parts = [];
  if (isInt(row.sp)) parts.push(`±${row.sp} 米`);
  if (isInt(row.sy)) parts.push(`±${row.sy}°`);
  return parts.join(' · ') || '—';
}

/** The count line: how many photos agreed, how many of them have a pose shown, why the rest are not. */
export function countsText(counts) {
  const c = counts || {};
  let s = `同意公开的 ${c.shared ?? 0} 张 / 共 ${c.photos ?? 0} 张`;
  if (c.shared) s += `，显示 ${c.shown ?? 0} 张`;
  if (c.not_determined) s += `；${c.not_determined} 张无法可靠确定位置和朝向，不显示`;
  return s;
}

/**
 * How often the ranges held, per source, from the server's measurement (uncertainty.measured_coverage): only the
 * sources with rows here, and "未实测" where nothing was measured.
 */
export function coverageText(uncertainty, rows) {
  const present = SOURCES.filter((s) => rows.some((r) => r.src === s));
  if (!present.length) return '';
  const measured = uncertainty?.measured_coverage ?? {};
  const sigmas = uncertainty?.sigmas ?? 2;
  const parts = present.map((s) => {
    const m = measured[s];
    if (!m || !m.n) return `${SOURCE_TEXT[s]}：未实测`;
    return `${SOURCE_TEXT[s]} ${m.n} 个中 ${m.within} 个（${Math.round((100 * m.within) / m.n)}%）`;
  });
  const runs = present.map((s) => measured[s]?.runs).find((r) => r > 1);
  return (
    `±范围是 ${sigmas}σ。在没有参与调参的测试地标上${runs ? `（${runs} 次运行合计）` : ''}，` +
    `真实位置和朝向都落在范围内的：${parts.join('；')}。`
  );
}

/** Which sources the rows have, explained. */
export function sourcesText(rows) {
  const present = SOURCES.filter((s) => rows.some((r) => r.src === s));
  return present.map((s) => `${SOURCE_TEXT[s]}：${SOURCE_HINT[s]}`).join('。') + (present.length ? '。' : '');
}
