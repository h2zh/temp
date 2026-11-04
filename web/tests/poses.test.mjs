// Run from the repo root: node --test 'web/tests/*.test.mjs'
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  SOURCE_TEXT,
  countsText,
  coverageText,
  headingText,
  markersKey,
  parsePoses,
  poseMarkers,
  positionText,
  rangeText,
  sourcesText,
} from '../poses.js';

const row = (over = {}) => ({ id: 'aaaaaaaaaaaa', src: 'sfm', p: [3, -4, 2], yaw: 10, sp: 16, sy: 21, ...over });
const doc = (photos, over = {}) => ({ format: 'crowd3d.photo-poses', version: 2, photos, counts: {}, ...over });

test('parsePoses accepts version 2 only and drops rows it cannot place, without rounding anything', () => {
  assert.equal(parsePoses(null), null);
  assert.equal(parsePoses(doc([], { version: 1 })), null);
  assert.equal(parsePoses(doc([], { format: 'other' })), null);
  assert.equal(parsePoses({ format: 'crowd3d.photo-poses', version: 2 }), null);
  const good = [row(), row({ id: 'b', src: 'exif', p: [-40, 7, null] }), row({ id: 'c', src: 'loc' })];
  const bad = [
    row({ src: 'gps', yaw: null }), // position only: never drawn
    row({ src: 'approx' }),
    row({ yaw: null }),
    row({ p: [1.4, 2, 0] }), // not whole metres: not a public row
    row({ p: [1, 2] }),
    row({ id: 7 }),
    null,
  ];
  const parsed = parsePoses(doc([...good, ...bad]));
  assert.deepEqual(parsed.photos, good);
});

test('markers stand at the row position, an EXIF row (no measured height) on the ground', () => {
  assert.deepEqual(poseMarkers([row(), row({ src: 'exif', p: [-40, 7, null], yaw: 92 })]), [
    { enu: [3, -4, 2], heading_deg: 10 },
    { enu: [-40, 7, 0], heading_deg: 92 },
  ]);
  assert.notEqual(markersKey([row()]), markersKey([row({ yaw: 11 })]));
  assert.equal(markersKey([]), '');
});

test('texts: position from the centre, heading with its direction, 2-sigma ranges', () => {
  assert.equal(positionText([12, -3, 2]), '东 12 米，南 3 米');
  assert.equal(positionText([-40, 7, null]), '西 40 米，北 7 米');
  assert.equal(positionText([0, 0, 0]), '东 0 米，北 0 米');
  assert.equal(headingText(0), '0°（北）');
  assert.equal(headingText(22), '22°（北）');
  assert.equal(headingText(23), '23°（东北）');
  assert.equal(headingText(183), '183°（南）');
  assert.equal(headingText(338), '338°（北）');
  assert.equal(rangeText(row()), '±16 米 · ±21°');
  assert.equal(rangeText(row({ sp: null, sy: null })), '—');
});

test('countsText says how many agreed, how many are shown and why the rest are not', () => {
  assert.equal(countsText({ photos: 9, shared: 0, shown: 0, not_determined: 0 }), '同意公开的 0 张 / 共 9 张');
  assert.equal(
    countsText({ photos: 9, shared: 4, shown: 3, not_determined: 1 }),
    '同意公开的 4 张 / 共 9 张，显示 3 张；1 张无法可靠确定位置和朝向，不显示',
  );
});

test('coverageText quotes per source the pooled share, the worst and best landmark-run and whether the rule was tuned there', () => {
  const u = {
    sigmas: 2,
    measured_coverage: {
      sfm: { within: 146, n: 165, runs: 2, landmarks: 4, lowest: { within: 1, n: 10 }, highest: { within: 13, n: 13 },
             rule_calibrated_here: false },
      loc: null,
      exif: { within: 23, n: 28, runs: 2, landmarks: 3, lowest: { within: 1, n: 2 }, highest: { within: 1, n: 1 },
              rule_calibrated_here: true },
    },
  };
  assert.equal(coverageText(u, []), '');
  assert.equal(
    coverageText(u, [row()]),
    '±范围是 2σ。在基准的测试地标上实测，真实位置和朝向都落在范围内的比例如下。' +
      '模型重建：4 个测试地标、2 次运行合计 165 个中 146 个（88%），单个地标单次运行最低 10 个中 1 个（10%）、' +
      '最高 13 个中 13 个（100%）；这个来源的范围规则没有用这些地标调过。',
  );
  const all = coverageText(u, [row(), row({ src: 'loc' }), row({ src: 'exif' })]);
  assert.match(all, /与模型比对：还没有实测。/);
  assert.match(
    all,
    /手机 GPS \+ 罗盘（未经模型核对）：3 个测试地标、2 次运行合计 28 个中 23 个（82%），单个地标单次运行最低 2 个中 1 个（50%）、最高 1 个中 1 个（100%）；这个来源的范围是用包括这些地标的照片定的，所以这个数不是独立检验，实际可能更低。$/,
  );
  // the EXIF figure is never called untuned or held out
  const exifOnly = coverageText(u, [row({ src: 'exif' })]);
  assert.doesNotMatch(exifOnly, /没有用这些地标调过|没有参与调参|留出/);
  // nothing claimed the server did not send: no spread, no calibration word, no landmark count
  const bare = coverageText({ measured_coverage: { sfm: { within: 3, n: 4 } } }, [row()]);
  assert.equal(bare, '±范围是 2σ。在基准的测试地标上实测，真实位置和朝向都落在范围内的比例如下。模型重建：测试地标 4 个中 3 个（75%）。');
  // a share the page cannot quote counts as not measured
  for (const bad of [{ within: 5, n: 4 }, { within: 1, n: 0 }, { within: 1.5, n: 4 }, { within: '3', n: 4 }]) {
    assert.match(coverageText({ measured_coverage: { sfm: bad } }, [row()]), /模型重建：还没有实测。$/);
  }
  assert.equal(SOURCE_TEXT.exif, '手机 GPS + 罗盘（未经模型核对）');
  assert.match(sourcesText([row({ src: 'exif' })]), /^手机 GPS \+ 罗盘（未经模型核对）：/);
  assert.equal(sourcesText([]), '');
});
