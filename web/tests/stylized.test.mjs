// Run from the repo root: node --test 'web/tests/*.test.mjs'
// The pure helpers of the stylized-model viewer (three.js is only imported inside main(), in a browser).
import assert from 'node:assert/strict';
import { test } from 'node:test';

globalThis.__STYLIZED_NO_AUTOSTART = true;
const {
  RELIEF_D24,
  RELIEF_D24_LABEL,
  RELIEF_KIND_TEXT,
  isReliefMeasured,
  structureLineSegments,
  AUTO_ADOPTED_LABEL,
  AUTO_SOURCE,
  DEFAULT_BEARING,
  GEOMETRY_TEXT,
  PROVENANCE_TEXT,
  SKIPPED_KINDS,
  TOON_STEPS,
  bearingOf,
  clayColour,
  describe,
  parseViewParams,
  pickLook,
  pickerChoices,
  preferredBearing,
  summarize,
  themeSearch,
  toonGradientData,
} = await import('../stylized.js');

const { DEFAULT_THEME } = await import('../themes.js');

const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('the toon ramp has one RGBA texel per step, rising to full brightness', () => {
  const data = toonGradientData();
  assert.equal(data.length, TOON_STEPS.length * 4);
  const reds = [...data].filter((_, i) => i % 4 === 0);
  assert.deepEqual(reds, [...reds].sort((a, b) => a - b));
  assert.equal(reds.at(-1), 255);
  assert.ok([...data].filter((_, i) => i % 4 === 3).every((a) => a === 255));
});

test('view parameters: defaults, clamping, and only same-origin model URLs', () => {
  const d = parseViewParams('?model=models/x.glb', 'https://crowd3d.example');
  assert.equal(d.model, '/models/x.glb');
  assert.equal(d.bearing, null);
  assert.equal(d.ui, true);
  const p = parseViewParams('?model=/m.glb&bearing=90&elev=200&zoom=abc&ui=0', 'https://crowd3d.example');
  assert.equal(p.bearing, 90);
  assert.equal(p.elevation, 85);
  assert.equal(p.zoom, 1);
  assert.equal(p.ui, false);
  assert.equal(parseViewParams('?model=https://evil.example/x.glb', 'https://crowd3d.example').model, null);
  assert.equal(parseViewParams('', 'https://crowd3d.example').model, null);
});

test('theme and outline parameters: the default look unless a valid theme id is given', () => {
  const d = parseViewParams('?model=/m.glb', 'https://crowd3d.example');
  assert.equal(d.theme, 'default');
  assert.equal(d.outlinePx, null); // the theme decides
  assert.equal(parseViewParams('?model=/m.glb&theme=dusk&outline=3', 'https://crowd3d.example').theme, 'dusk');
  assert.equal(parseViewParams('?model=/m.glb&outline=3').outlinePx, 3);
  for (const bad of ['../secret', 'Dusk', 'a b', '', 'x'.repeat(41)]) {
    assert.equal(parseViewParams(`?model=/m.glb&theme=${encodeURIComponent(bad)}`).theme, 'default', bad);
  }
});

test('the theme picker: index order, the current theme kept, other parameters kept', () => {
  assert.deepEqual(pickerChoices({ themes: ['default', 'storybook', 'dusk'] }, 'dusk'), ['default', 'storybook', 'dusk']);
  assert.deepEqual(pickerChoices({ themes: ['storybook'] }, 'mine'), ['default', 'storybook', 'mine']);
  assert.deepEqual(pickerChoices(null, 'default'), ['default']);
  assert.deepEqual(pickerChoices({ themes: ['../x', 'dusk'] }, '../y'), ['default', 'dusk']);
  assert.equal(themeSearch('?model=%2Fm.glb&bearing=90', 'dusk'), '?model=%2Fm.glb&bearing=90&theme=dusk');
  assert.equal(themeSearch('?model=%2Fm.glb&theme=dusk', 'default'), '?model=%2Fm.glb');
  assert.equal(themeSearch('?theme=dusk', 'default'), '');
});

test('compass bearings in three.js axes (x east, -z north)', () => {
  near(bearingOf(0, -1), 0);
  near(bearingOf(1, 0), 90);
  near(bearingOf(0, 1), 180);
  near(bearingOf(-1, 0), 270);
});

test('the first view faces the photographed walls, weighted by area', () => {
  const walls = [
    { normal: [0, 0, 1], area: 100, observed: true }, // facing south
    { normal: [1, 0, 0], area: 20, observed: true }, // east, smaller
    { normal: [0, 0, -1], area: 500, observed: false }, // north, not photographed: ignored
  ];
  const b = preferredBearing(walls);
  assert.ok(b > 160 && b < 180, String(b));
  // no photographed wall (every model whose photos were not aligned): the fixed default view
  assert.equal(DEFAULT_BEARING, 140);
  assert.equal(preferredBearing([{ normal: [0, 0, 1], area: 5, observed: false }]), DEFAULT_BEARING);
  assert.equal(preferredBearing([], 200), 200);
});

test('every face is clay, painted by its kind: roofs in clay.roof, everything else in clay.wall', () => {
  const clay = { wall: '#111111', roof: '#222222' };
  assert.equal(clayColour(clay, 'roof'), '#222222');
  for (const kind of ['wall', 'corners', 'base', 'roof-cut', undefined]) assert.equal(clayColour(clay, kind), '#111111', String(kind));
  // the photo-coloured roof decals of older models are not drawn (the roof face is whole underneath)
  assert.deepEqual([...SKIPPED_KINDS], ['roof-patch']);
  // the default look keeps the clay every clay wall and roof showed before D29 (the old clay.face and clay.roof)
  const { look } = pickLook(DEFAULT_THEME, 'default', []);
  assert.equal(clayColour(look.clay, 'wall'), '#c8d1df');
  assert.equal(clayColour(look.clay, 'roof'), '#c9d2e2');
  assert.equal(look.clay.hatch_shade, 0.84);
});

test('panel counts and the provenance shown for a tapped face', () => {
  const extras = [
    // wall 0 split into two nodes (one wall)
    { kind: 'wall', wall: 0, provenance: { outline: 'osm-footprint', height: 'sfm-points' } },
    { kind: 'wall', wall: 0, provenance: { outline: 'osm-footprint', height: 'sfm-points' } },
    { kind: 'wall', wall: 1, provenance: { outline: 'osm-footprint', height: 'inferred' } },
    // a pyramid roof in two nodes
    { kind: 'roof', faces: [0, 1, 2], area_m2: 60, provenance: { outline: 'osm-footprint', height: 'sfm-points' } },
    { kind: 'roof', faces: [3, 4, 5], area_m2: 40, provenance: { outline: 'osm-footprint', height: 'default-style' } },
    { kind: 'corners' },
  ];
  assert.deepEqual(summarize(extras), { walls: 2, roofFaces: 6, withoutProvenance: 1, reliefElements: 0 });
  const lines = describe(extras[2]);
  assert.equal(lines[0], '立面 #1');
  assert.match(lines[1], /OSM 轮廓/);
  assert.match(lines[2], /推断/);
  assert.equal(lines.length, 3);
  assert.match(describe(extras[4])[2], /默认样式/);
  assert.doesNotMatch(describe(extras[0])[2], /取自/);
  // models written before 'inferred' existed
  assert.match(describe({ ...extras[1], height_measured_here: false })[2], /重建点云实测（取自同一体块已测的墙）/);
  assert.deepEqual(describe({}), []);
});

test('an older GLB with photo colours reads as clay: no colour, coverage, photo or unlock lines', () => {
  // the extras crowd3d.stylize wrote before D29 on a photographed wall strip and an observed roof
  const wall = { kind: 'wall', wall: 0, observed: true, locked: false, observed_fraction: 0.8, base_color: [200, 180, 150],
    photos: [{ photo: 'a', share: 0.6 }, { photo: 'b', share: 0.4 }], fill: { holes: 2, area_m2: 0.43, px_per_m: 16 },
    provenance: { outline: 'osm-footprint', height: 'sfm-points', appearance: 'photo-observed' } };
  const roof = { kind: 'roof', faces: [0, 1], area_m2: 10, observed: true, locked: false, coverage: [0.7, 0.9],
    provenance: { outline: 'osm-footprint', height: 'sfm-points', appearance: 'photo-observed' } };
  const shown = [...describe(wall), ...describe(roof)].join(' | ');
  for (const re of [/外观/, /照片实拍/, /实拍/, /覆盖/, /补洞/, /张照片/, /解锁/]) assert.doesNotMatch(shown, re);
  assert.deepEqual(describe(wall), ['立面 #0', '轮廓：OSM 轮廓', '高度：重建点云实测']);
  assert.deepEqual(describe(roof), ['屋顶', '轮廓：OSM 轮廓', '高度：重建点云实测']);
  assert.deepEqual(summarize([wall, roof]), { walls: 1, roofFaces: 2, withoutProvenance: 0, reliefElements: 0 });
  for (const v of ['photo-observed', 'observed-fill']) assert.ok(!(v in PROVENANCE_TEXT), v);
});

test('a GLB written by crowd3d.stylize passes the Khronos glTF validator (skipped without it)', async (t) => {
  // The validator is not a repo dependency: point GLTF_VALIDATOR at an installed gltf-validator package
  // directory and STYLIZED_GLB at a model (e.g. one written by scripts/stylize_model.py); otherwise skip.
  const dir = process.env.GLTF_VALIDATOR;
  const glb = process.env.STYLIZED_GLB;
  if (!dir || !glb) return t.skip('set GLTF_VALIDATOR and STYLIZED_GLB to run');
  const { createRequire } = await import('node:module');
  const { readFileSync } = await import('node:fs');
  const validator = createRequire(import.meta.url)(dir);
  const report = await validator.validateBytes(new Uint8Array(readFileSync(glb)), { maxIssues: 100 });
  assert.equal(report.issues.numErrors, 0, JSON.stringify(report.issues.messages.slice(0, 5)));
  assert.equal(report.issues.numWarnings, 0, JSON.stringify(report.issues.messages.slice(0, 5)));
});

test('a structure part says where its shape comes from', () => {
  const ex = { kind: 'roof', part: 'w505513556', geometry: 'map-convention', observed: false, locked: true,
    roof: { shape: 'flat', fallback_from: 'pyramidal' },
    provenance: { outline: 'osm-s3db', height: 's3db-shape-default', appearance: 'default-style' } };
  const lines = describe(ex);
  assert.match(lines[1], /OpenStreetMap 志愿者测绘的分件/);
  assert.match(lines[2], /屋顶坡度按惯例/);
  assert.ok(lines.some((l) => /地图事实加惯例/.test(l)));
  assert.ok(lines.some((l) => /pyramidal.*flat/.test(l)));
  assert.match(describe({ kind: 'base', locked: true, provenance: {} })[0], /底面/);
  for (const v of Object.keys(PROVENANCE_TEXT)) assert.ok(PROVENANCE_TEXT[v].length > 0);
});

test('a height-pending stub reads as one, even when a model calls its geometry disputed', () => {
  // out-new/aachen-cathedral.glb: the judge had turned stub remainder13 into 'disputed', which read as an OSM-mapped shape
  const ex = { kind: 'wall', wall: 151, part: 'remainder13', geometry: 'disputed', observed: false, locked: true,
    provenance: { outline: 'osm-footprint', height: 'default-style', appearance: 'default-style' } };
  const lines = describe(ex);
  assert.ok(lines.some((l) => /高度待定/.test(l)), lines.join(' | '));
  assert.ok(!lines.some((l) => /^形状：/.test(l)), lines.join(' | '));
  // a mapped part keeps its own geometry text
  const mapped = describe({ ...ex, part: 'w505513556', geometry: 'disputed', provenance: { ...ex.provenance, height: 'osm-s3db' } });
  assert.ok(mapped.includes(GEOMETRY_TEXT.disputed) && !mapped.some((l) => /高度待定/.test(l)));
});

test('a value two photo channels agreed on reads 照片量出（自动采用） with the map value beside it (D15)', () => {
  // the shape crowd3d.stylize writes for a part whose top D15 adopted, and for a part carried with it
  const ex = { kind: 'wall', wall: 3, part: 'w2', geometry: 'photo-fitted', observed: false, locked: true,
    adopted: [{ param: 'height', map: 24, value: 30.01 }],
    provenance: { outline: 'osm-s3db', height: AUTO_SOURCE, appearance: 'default-style' } };
  const lines = describe(ex);
  assert.equal(AUTO_ADOPTED_LABEL, '照片量出（自动采用）');
  assert.equal(PROVENANCE_TEXT[AUTO_SOURCE], AUTO_ADOPTED_LABEL);
  assert.ok(lines.includes(`高度：${AUTO_ADOPTED_LABEL}`), lines.join(' | '));
  assert.ok(lines.includes(`形状：${AUTO_ADOPTED_LABEL}`), lines.join(' | '));
  assert.ok(lines.some((l) => /顶高：照片量出（自动采用） 30\.01 m（地图原值 24 m，可撤销）/.test(l)), lines.join(' | '));
  assert.ok(!lines.includes(GEOMETRY_TEXT['photo-fitted']), 'never the curator-confirmed label');
  assert.ok(!lines.some((l) => /外观|照片实拍|解锁/.test(l)), 'the adoption is geometry only: nothing about colour');
  const carried = describe({ ...ex, part: 'w3', adopted: [{ param: 'min_height', map: 24, value: 30, carried: true }] });
  assert.ok(carried.some((l) => /底高：随下面的分件一起移动 30 m（地图原值 24 m，可撤销）/.test(l)), carried.join(' | '));
  // an adopted outline (the frame, D15 ⑥) reads as photo-measured, not as OpenStreetMap's
  const framed = describe({ ...ex, geometry: 'map', adopted: undefined,
    provenance: { outline: AUTO_SOURCE, height: 'osm-s3db', appearance: 'default-style' } });
  assert.ok(framed.includes(`轮廓：${AUTO_ADOPTED_LABEL}`), framed.join(' | '));
  // ... and its shape line says so too (never a bare "OpenStreetMap" beside it), with the map's old outline from
  // the node extras crowd3d.stylize writes (outline_adopted)
  assert.ok(!framed.includes(GEOMETRY_TEXT.map), framed.join(' | '));
  assert.ok(framed.includes(`${GEOMETRY_TEXT.map}（轮廓按照片整体缩放、平移：${AUTO_ADOPTED_LABEL}）`), framed.join(' | '));
  const framed2 = describe({ ...ex, geometry: 'map', adopted: undefined,
    outline_adopted: { adoption: 'A-x-1', k: 0.8178, t: [3.351, -1.175], width_map_m: 33.786, width_m: 27.631 },
    provenance: { outline: AUTO_SOURCE, height: 'osm-s3db', appearance: 'default-style' } });
  assert.ok(framed2.includes(`轮廓：${AUTO_ADOPTED_LABEL}：整栋缩放 ×0.818、平移 3.55 m（地图原轮廓宽 33.79 m，现 27.63 m，可撤销）`),
    framed2.join(' | '));
  // a stub that got no measured height still reads as one
  const stub = describe({ ...ex, adopted: undefined, geometry: 'photo-fitted',
    provenance: { ...ex.provenance, height: 'default-style' } });
  assert.ok(stub.some((l) => /高度待定/.test(l)) && !stub.some((l) => /自动采用/.test(l.replace(/^高度：.*/, ''))));
});

test('a value strong photo evidence corrected reads 照片纠正地图 with old -> new (D25)', () => {
  // the node extras crowd3d.stylize writes for a D25 correction (crowd3d.correct): the map's value, the adopted value,
  // what was adopted (the photo value or only its bound) and the registration box's range
  const ex = { kind: 'wall', wall: 5, part: 'w505513556', geometry: 'photo-corrected', observed: false, locked: true,
    corrected: [{ param: 'height', map: 32, value: 46.92, adopted: 'measured', record: 'D25-a-1', kind: 'd25-height',
      range: [42.17, 53.67], photo_value: 46.92 }],
    provenance: { outline: 'osm-s3db', height: 'photo-corrected (auto)', appearance: 'default-style' } };
  const lines = describe(ex);
  assert.ok(lines.includes('高度：照片纠正地图（自动）'), lines.join(' | '));
  assert.ok(lines.includes('形状：照片纠正了地图'), lines.join(' | '));
  assert.ok(lines.includes('照片纠正地图（自动）：顶高 地图 32 m → 照片 46.92 m（范围 42.17 m–53.67 m）（可撤销）'), lines.join(' | '));
  const lb = describe({ ...ex, provenance: { ...ex.provenance, height: 'photo-corrected (auto, lower bound)' },
    corrected: [{ ...ex.corrected[0], adopted: 'lower bound', value: 38.1 }] });
  assert.ok(lb.includes('高度：照片纠正地图（自动，下界）'), lb.join(' | '));
  assert.ok(lb.includes('照片纠正地图（自动，下界）：顶高至少 38.1 m（地图 32 m，照片值 46.92 m，可撤销）'), lb.join(' | '));
  const block = describe({ ...ex, part: 'remainder13', corrected: [{ param: 'height', map: 3, value: 16.24, adopted: 'lower bound',
    kind: 'd25-block', record: 'D25-a-3' }] });
  assert.ok(block.includes('照片量出：至少 16.24 m（屋顶形状未知；地图上高度待定 3 m，可撤销）'), block.join(' | '));
  const carried = describe({ ...ex, part: 'w9', corrected: [{ param: 'min_height', map: 32, value: 46.92, carried: true }] });
  assert.ok(carried.includes('底高：随下面被纠正的分件一起移动 46.92 m（地图原值 32 m，可撤销）'), carried.join(' | '));
  const tilt = describe({ ...ex, corrected: undefined, geometry: 'map', tilt: { tilt_deg: 4.11, azimuth_deg: 183.0 },
    provenance: { ...ex.provenance, height: 'osm-s3db' } });
  assert.ok(tilt.includes('倾斜：照片量出 4.1°，朝 183°（地图上是直的，可撤销）'), tilt.join(' | '));
  assert.ok(!tilt.some((l) => /照片纠正地图（自动）：/.test(l)));
  // one decimal always (verify-r1: 4.0117 read '4°')
  const t4 = describe({ ...ex, corrected: undefined, geometry: 'map', tilt: { tilt_deg: 4.0117, azimuth_deg: 177.677 },
    provenance: { ...ex.provenance, height: 'osm-s3db' } });
  assert.ok(t4.includes('倾斜：照片量出 4.0°，朝 178°（地图上是直的，可撤销）'), t4.join(' | '));
});

test('the default look passes the validator; a theme that fails it gives way to the default, with a note', () => {
  const notes = [];
  const ok = pickLook(DEFAULT_THEME, 'default', notes);
  assert.ok(ok.check.ok && notes.length === 0);
  assert.equal(ok.look.id, 'default');
  // a theme of the old colour schema (palette rules), and one whose hatch cannot show
  const old = { ...structuredClone(DEFAULT_THEME), id: 'old', name: '旧画风', palette: { rules: [{ match: '*' }] } };
  const flat = structuredClone(DEFAULT_THEME);
  Object.assign(flat, { id: 'flat', name: '没纹理' });
  flat.clay.hatch_shade = 0.99;
  for (const [bad, re] of [[old, /旧画风.*没有通过校验.*unknown key "palette"/], [flat, /没纹理.*没有通过校验.*hatch_shade/]]) {
    const n = [];
    const { look, check } = pickLook(bad, bad.id, n);
    assert.equal(look.id, 'default');
    assert.ok(check.ok);
    assert.equal(n.length, 1);
    assert.match(n[0], re);
  }
  const n = [];
  assert.equal(pickLook(null, 'x', n).look.id, 'default');
  assert.match(n[0], /画风“x”没有通过校验/);
});

test('a photographed wall lists the relief the photos measured on it, and only photo-measured records count', () => {
  const rec = (kind, built, provenance = 'photo-measured') => ({ type: 'opening', kind, built, provenance });
  const wall = { kind: 'wall', wall: 3, observed: true, locked: false,
    provenance: { outline: 'osm-s3db', height: 'osm-s3db', appearance: 'photo-observed' },
    relief: [rec('arched-bay', true), rec('arched-bay', true), rec('setback', true), rec('pilaster', false), rec('door', true, 'inferred')] };
  const lines = describe(wall);
  const main = lines.find((l) => /立面起伏/.test(l));
  assert.match(main, /照片实测 3 处（拱券开间 2、退台 1）/);
  assert.ok(lines.some((l) => /另实测 1 处不单独建模/.test(l)));
  assert.ok(!lines.some((l) => /门洞/.test(l)));                       // an unbacked record is never shown as measured
  assert.ok(!describe({ ...wall, relief: [] }).some((l) => /立面起伏/.test(l)));
  assert.equal(summarize([wall]).reliefElements, 3);
  for (const v of Object.values(RELIEF_KIND_TEXT)) assert.ok(v.length > 0);
});

test('relief built on the point cloud alone (D24) says so, and is never shown as checked by 3-D lines', () => {
  const rec = (kind, built, provenance, d24 = { scale_condition: { allowed: true } }) => ({ type: 'opening', kind, built, provenance, d24 });
  const wall = { kind: 'wall', wall: 2, observed: true, locked: false,
    provenance: { outline: 'osm-s3db', height: 'osm-s3db', appearance: 'photo-observed' },
    relief: [rec('arched-bay', true, RELIEF_D24), rec('arched-bay', true, RELIEF_D24), rec('setback', true, 'photo-measured', undefined),
      rec('pilaster', false, RELIEF_D24), rec('bay', true, RELIEF_D24, null)] };
  const lines = describe(wall);
  const d24 = lines.find((l) => l.includes(RELIEF_D24_LABEL));
  assert.equal(RELIEF_D24_LABEL, '照片点云量出（D24）');
  assert.match(d24, /照片点云量出（D24） 2 处（拱券开间 2）/);
  assert.match(d24, /偶然水平/);
  assert.ok(!/并有三维直线核对/.test(d24));
  assert.ok(lines.some((l) => /照片实测 1 处（退台 1），每处两半照片各自量到，并有三维直线核对/.test(l)));
  assert.ok(lines.some((l) => /另实测 1 处不单独建模/.test(l)));
  assert.ok(!d24.includes('凹开间'));                                  // a D24 record without its evidence is not shown
  assert.equal(summarize([wall]).reliefElements, 3);
  assert.ok(isReliefMeasured(rec('bay', true, RELIEF_D24)) && !isReliefMeasured(rec('bay', true, RELIEF_D24, null)));
  assert.ok(!isReliefMeasured({ provenance: 'inferred' }) && !isReliefMeasured(null));
});

test('the roof lines a model lists are drawn solid, a conventional roof shape dashed, eaves and malformed rows skipped', () => {
  const sl = { types: ['ridge', 'hip', 'valley', 'eave', 'rake'],
    lines: [[0, 10, 0, 5, 14, -5, 1, 0, 1, 0], [1, 2, 3, 4, 5, 6, 0, 1, 0, 1], [1, 2, 'x', 4, 5, 6, 0, 0, 0, 0], 'junk',
      [7, 7, 7, 8, 8, 8, 3, 0, 0, 0]] };
  const { solid, dashed } = structureLineSegments(sl);
  assert.deepEqual(solid, [0, 10, 0, 5, 14, -5]);
  assert.deepEqual(dashed, [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(structureLineSegments(undefined), { solid: [], dashed: [] });
});
