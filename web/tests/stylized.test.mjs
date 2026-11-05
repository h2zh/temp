// Run from the repo root: node --test 'web/tests/*.test.mjs'
// The pure helpers of the stylized-model viewer (three.js is only imported inside main(), in a browser).
import assert from 'node:assert/strict';
import { test } from 'node:test';

globalThis.__STYLIZED_NO_AUTOSTART = true;
const {
  RELIEF_D24,
  RELIEF_KIND_TEXT,
  isReliefMeasured,
  structureLineSegments,
  autoAdoptedLabel,
  reliefD24Label,
  AUTO_SOURCE,
  DEFAULT_BEARING,
  GEOMETRY_TEXT,
  PHOTO_ATTRIBUTION,
  PROVENANCE_TEXT,
  SKIPPED_KINDS,
  TOON_STEPS,
  bearingOf,
  clayColour,
  describe,
  fillPanel,
  parseViewParams,
  pickLook,
  pickerChoices,
  preferredBearing,
  summarize,
  themeSearch,
  toonGradientData,
} = await import('../stylized.js');
// the labels in the page language (node: Chinese, the language the pages are written in)
const AUTO_ADOPTED_LABEL = autoAdoptedLabel();
const RELIEF_D24_LABEL = reliefD24Label();

const { DEFAULT_THEME, loadTheme, themeName } = await import('../themes.js');
const { missing } = await import('../i18n.js');
const { STRINGS } = await import('../strings.js');
const { inLang, installDom, text } = await import('./fake-dom.mjs');
const { readFileSync, readdirSync } = await import('node:fs');

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

test('every face is default-style since D29: the text never says a photographed wall was not photographed', () => {
  // crowd3d.stylize now writes appearance default-style on every node; `observed` alone marks a photographed wall
  assert.doesNotMatch(PROVENANCE_TEXT['default-style'], /拍/);
  const wall = { kind: 'wall', wall: 0, observed: true,
    provenance: { outline: 'osm-footprint', height: 'default-style', appearance: 'default-style' } };
  assert.deepEqual(describe(wall), ['立面 #0', '轮廓：OSM 轮廓', '高度：默认样式（没有实测，也没有标注）']);
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
  const wall = { kind: 'wall', wall: 3, observed: true,
    provenance: { outline: 'osm-s3db', height: 'osm-s3db', appearance: 'default-style' },
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
  const wall = { kind: 'wall', wall: 2, observed: true,
    provenance: { outline: 'osm-s3db', height: 'osm-s3db', appearance: 'default-style' },
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

// ---------- English (D32) ----------

// Chinese characters and CJK / full-width punctuation
const CJK = /[\p{Script=Han}　-〿＀-￯]/u;
const noCjk = (lines, what) => {
  for (const l of [].concat(lines)) assert.ok(!CJK.test(l), `${what}: ${l}`);
};

test('in English: the panel, its counts and notes; the GLB attribution as it comes (D32)', () => {
  inLang('en', () => {
    const doc = installDom();
    try {
      fillPanel(doc, { landmark: undefined, stats: { walls: 12, roofFaces: 1, reliefElements: 3 }, notes: ['a', 'b'],
        attribution: ['© OpenStreetMap contributors, ODbL'] });
      assert.equal(text(doc.getElementById('title')), 'Landmark');
      assert.equal(text(doc.getElementById('walls')), '12 walls');
      assert.equal(text(doc.getElementById('roof-faces')), '1 roof face');
      assert.equal(text(doc.getElementById('relief-chip')), '3 relief features');
      assert.equal(doc.getElementById('relief-chip').hidden, false);
      assert.equal(text(doc.getElementById('theme-note')), 'a; b');
      assert.equal(doc.getElementById('theme-note').hidden, false);
      assert.equal(text(doc.getElementById('attribution')), '© OpenStreetMap contributors, ODbL');
      noCjk(['title', 'walls', 'roof-faces', 'relief-chip'].map((id) => text(doc.getElementById(id))), 'panel');
    } finally {
      doc.restore();
    }
    const doc2 = installDom();
    try {
      // a landmark's name is shown as the model gives it, never translated
      fillPanel(doc2, { landmark: '比萨斜塔', stats: { walls: 1, roofFaces: 0, reliefElements: 0 } });
      assert.equal(text(doc2.getElementById('title')), '比萨斜塔');
      assert.equal(doc2.getElementById('relief-chip').hidden, true);
      assert.equal(doc2.getElementById('theme-note').hidden, true);
    } finally {
      doc2.restore();
    }
  });
  // the product's own attribution line in the page language; the licence lines as they come
  const glbLines = ['© OpenStreetMap contributors, ODbL', PHOTO_ATTRIBUTION];
  for (const [l, want] of [
    ['zh', '摆放位置和照片量出的细节来自这个地标的打卡照片（Crowd3D 共同完成者）'],
    ['en', "Placement and photo-measured detail from this landmark's check-in photos (Crowd3D contributors)"],
  ]) {
    inLang(l, () => {
      const doc3 = installDom();
      try {
        fillPanel(doc3, { landmark: 'X', stats: { walls: 1, roofFaces: 0, reliefElements: 0 }, attribution: glbLines });
        assert.deepEqual(doc3.getElementById('attribution').children.map(text), ['© OpenStreetMap contributors, ODbL', want], l);
      } finally {
        doc3.restore();
      }
    });
  }
  // the line is the one the stylize job writes into the GLB
  const pipeline = readFileSync(new URL('../../crowd3d/structure_pipeline.py', import.meta.url), 'utf8');
  assert.equal(pipeline.match(/^PHOTO_ATTRIBUTION = "([^"]*)"$/m)?.[1], PHOTO_ATTRIBUTION);
  // the panel's paragraph: a clay model, never coloured, as in Chinese (不上色)
  assert.match(STRINGS['stylized.about'].zh, /不上色/);
  assert.match(STRINGS['stylized.about'].en, /^The clay model is built from the map's outlines and heights;.* It is not coloured\./);
  assert.match(STRINGS['stylized.metaDescription'].en, /clay model .*Not coloured\.$/);
});

test('in English: the card of a tapped wall and roof says where each value comes from, and never claims colour (D32)', () => {
  inLang('en', () => {
    assert.deepEqual(describe({ kind: 'wall', wall: 1, provenance: { outline: 'osm-footprint', height: 'inferred' } }),
      ['Wall #1', 'Outline: OSM outline', 'Height: Inferred (measured elsewhere on the same block, not here)']);
    assert.deepEqual(describe({ kind: 'roof', faces: [0, 1], provenance: { outline: 'osm-footprint', height: 'default-style' } }),
      ['Roof', 'Outline: OSM outline', 'Height: Default style (neither measured nor tagged)']);
    assert.deepEqual(describe({ kind: 'wall', wall: 0, height_measured_here: false, provenance: { outline: 'osm-footprint', height: 'sfm-points' } }),
      ['Wall #0', 'Outline: OSM outline', 'Height: Measured on the reconstructed point cloud (from a measured wall of the same block)']);
    assert.deepEqual(describe({ kind: 'base', provenance: {} }), ['Base', 'Outline: Unknown', 'Height: Unknown']);
    assert.equal(describe({ kind: 'roof-cut', provenance: {} })[0], 'Roof cut');
    assert.equal(describe({ kind: 'corners', provenance: {} })[0], 'Corner');
    // the default style is neither measured nor tagged: nothing about photos or colour, as the Chinese says nothing of 拍
    assert.doesNotMatch(PROVENANCE_TEXT['default-style'], /photo|colou?r|paint|textur/i);
    const photographed = describe({ kind: 'wall', wall: 0, observed: true,
      provenance: { outline: 'osm-footprint', height: 'default-style', appearance: 'default-style' } });
    assert.deepEqual(photographed, ['Wall #0', 'Outline: OSM outline', 'Height: Default style (neither measured nor tagged)']);
    // an older GLB with photo colours reads as clay: no colour, coverage, fill or unlock lines
    const wall = { kind: 'wall', wall: 0, observed: true, locked: false, observed_fraction: 0.8, base_color: [200, 180, 150],
      photos: [{ photo: 'a', share: 0.6 }], fill: { holes: 2, area_m2: 0.43, px_per_m: 16 },
      provenance: { outline: 'osm-footprint', height: 'sfm-points', appearance: 'photo-observed' } };
    const shown = describe(wall).join(' | ');
    assert.equal(shown, 'Wall #0 | Outline: OSM outline | Height: Measured on the reconstructed point cloud');
    assert.doesNotMatch(shown, /colou?r|appearance|coverage|fill|unlock|photographed/i);
    // a structure part: its shape and a roof the map tags but that could not be built
    const part = describe({ kind: 'roof', part: 'w505513556', geometry: 'map-convention', roof: { shape: 'flat', fallback_from: 'pyramidal' },
      provenance: { outline: 'osm-s3db', height: 's3db-shape-default', appearance: 'default-style' } });
    assert.deepEqual(part, [
      'Roof',
      'Outline: Mapped by OpenStreetMap volunteers (building part)',
      'Height: Conventional roof pitch (the map only gives the roof shape)',
      'Shape: map facts plus convention (conventional roof pitch)',
      'Roof: the map tags it pyramidal, which could not be built, so it is drawn flat',
    ]);
    const stub = describe({ kind: 'wall', wall: 151, part: 'remainder13', geometry: 'disputed', provenance: { outline: 'osm-footprint', height: 'default-style' } });
    assert.ok(stub.includes('Height pending (the map has no height)') && !stub.some((l) => /^Shape:/.test(l)), stub.join(' | '));
  });
});

test('in English: values the photos changed (D15, D25) and relief the photos measured (D24) keep the map value and say how (D32)', () => {
  inLang('en', () => {
    assert.equal(autoAdoptedLabel(), 'Measured from photos (adopted automatically)');
    assert.equal(PROVENANCE_TEXT[AUTO_SOURCE], autoAdoptedLabel());
    const d15 = describe({ kind: 'wall', wall: 3, part: 'w2', geometry: 'photo-fitted', adopted: [{ param: 'height', map: 24, value: 30.01 }],
      outline_adopted: { k: 0.8178, t: [3.351, -1.175], width_map_m: 33.786, width_m: 27.631 },
      provenance: { outline: AUTO_SOURCE, height: AUTO_SOURCE, appearance: 'default-style' } });
    assert.deepEqual(d15, [
      'Wall #3',
      'Outline: Measured from photos (adopted automatically)',
      'Height: Measured from photos (adopted automatically)',
      'Shape: Measured from photos (adopted automatically) (outline scaled and shifted as a whole to fit the photos: Measured from photos (adopted automatically))',
      'Outline: Measured from photos (adopted automatically): the whole building scaled ×0.818 and shifted 3.55 m (the map outline was 33.79 m wide, now 27.63 m; can be undone)',
      'Top height: Measured from photos (adopted automatically), 30.01 m (map value 24 m; can be undone)',
    ]);
    const carried = describe({ kind: 'wall', part: 'w3', geometry: 'photo-fitted', adopted: [{ param: 'min_height', map: 24, value: 30, carried: true }],
      provenance: { outline: 'osm-s3db', height: AUTO_SOURCE } });
    assert.ok(carried.includes('Base height: moved with the part below, 30 m (map value 24 m; can be undone)'), carried.join(' | '));
    const d25 = describe({ kind: 'wall', wall: 5, part: 'w5', geometry: 'photo-corrected',
      corrected: [
        { param: 'height', map: 32, value: 46.92, adopted: 'measured', range: [42.17, 53.67] },
        { param: 'roof_height', map: 32, value: 38.1, adopted: 'lower bound', photo_value: 46.92 },
        { param: 'height', map: 3, value: 16.24, adopted: 'lower bound', kind: 'd25-block' },
        { param: 'min_height', map: 32, value: 46.92, carried: true },
      ],
      tilt: { tilt_deg: 4.0117, azimuth_deg: 177.677 },
      provenance: { outline: 'osm-s3db', height: 'photo-corrected (auto)' } });
    assert.deepEqual(d25.slice(2), [
      'Height: Photos corrected the map (automatically)',
      'Shape: photos corrected the map',
      'Photos corrected the map (automatically): Top height 32 m on the map → 46.92 m from photos (range 42.17 m–53.67 m; can be undone)',
      'Photos corrected the map (automatically, lower bound): Roof height at least 38.1 m (map 32 m, photo value 46.92 m; can be undone)',
      'Measured from photos: at least 16.24 m (roof shape unknown; height pending on the map, 3 m; can be undone)',
      'Base height: moved with the corrected part below, 46.92 m (map value 32 m; can be undone)',
      'Lean: 4.0° towards 178°, measured from photos (upright on the map; can be undone)',
    ]);
    // relief: with a 3-D line, on the point cloud alone (D24), and measured but not built; plurals per kind
    const rec = (kind, built, provenance, d24 = { scale_condition: { allowed: true } }) => ({ type: 'opening', kind, built, provenance, d24 });
    const relief = describe({ kind: 'wall', wall: 2, provenance: { outline: 'osm-s3db', height: 'osm-s3db' },
      relief: [rec('arched-bay', true, RELIEF_D24), rec('arched-bay', true, RELIEF_D24), rec('setback', true, 'photo-measured', undefined),
        rec('buttress', true, 'photo-measured'), rec('buttress', true, 'photo-measured'), rec('pilaster', false, RELIEF_D24)] });
    assert.equal(reliefD24Label(), 'measured on the photo point cloud alone (D24)');
    assert.deepEqual(relief.slice(3), [
      'Relief measured by photos: 3 features (1 setback and 2 buttresses), each found separately in both halves of the photos and checked against a 3-D line',
      'Relief measured on the photo point cloud alone (D24): 2 features (2 arched bays). Both ways of splitting the photos found them and re-measuring the raw points agrees; this wall has 3-D lines everywhere (at chance level), so they were not checked against a line',
      '1 more feature measured but not modelled on its own (a pilaster between two recessed bays, for example, is the wall itself)',
    ]);
    assert.doesNotMatch(relief[4], /checked against a 3-D line/);
    // every code the card knows, in every combination the card builds: no Chinese
    const codes = [...Object.keys(PROVENANCE_TEXT), 'unknown-code', undefined];
    for (const kind of ['wall', 'roof', 'roof-cut', 'corners', 'base']) {
      for (const outline of codes) for (const height of codes) noCjk(describe({ kind, provenance: { outline, height } }), `${kind} ${outline} ${height}`);
    }
    for (const geometry of [...Object.keys(GEOMETRY_TEXT), 'photo-fitted']) {
      noCjk(describe({ kind: 'wall', part: 'p', geometry, outline_adopted: true, provenance: { outline: AUTO_SOURCE, height: AUTO_SOURCE } }), geometry);
    }
    for (const kind of Object.keys(RELIEF_KIND_TEXT)) {
      noCjk(describe({ kind: 'wall', relief: [rec(kind, true, 'photo-measured'), rec(kind, true, RELIEF_D24), rec(kind, true, RELIEF_D24), rec(kind, false, RELIEF_D24)] }), kind);
    }
  });
});

test('in English: the style picker shows the English names; notes about a style that cannot be used (D32)', async () => {
  const dir = new URL('../themes/', import.meta.url);
  const themes = readdirSync(dir).filter((f) => f.endsWith('.json') && f !== 'index.json').map((f) => JSON.parse(readFileSync(new URL(f, dir), 'utf8')));
  inLang('en', () => {
    assert.equal(themeName(DEFAULT_THEME), 'Bright flat');
    for (const th of themes) {
      assert.equal(themeName(th), th.en.name);
      noCjk(themeName(th), th.id);
    }
    const n = [];
    pickLook({ ...structuredClone(DEFAULT_THEME), id: 'old', name: '旧画风', en: { name: 'Old style', description: 'Old.' }, palette: {} }, 'old', n);
    pickLook({ id: 'nameless', name: '无名' }, 'nameless', n);
    assert.deepEqual(n, [
      'The style "Old style" failed validation, so the default style is used: unknown key "palette"',
      'The style "nameless" failed validation, so the default style is used: schema: must be 1',
    ]);
  });
  // Chinese: the Chinese names
  assert.equal(themeName(DEFAULT_THEME), '明快扁平');
  for (const th of themes) assert.equal(themeName(th), th.name);
  const fetchJson = async (url) => {
    if (url === 'themes/odd.json') return { id: 'other' };
    throw new Error('HTTP 404');
  };
  await inLang('en', async () => {
    assert.equal((await loadTheme('../secret', fetchJson)).error, 'Not a valid style name: ../secret');
    assert.equal((await loadTheme('odd', fetchJson)).error, 'The style file odd.json does not have the id odd');
    assert.equal((await loadTheme('missing', fetchJson)).error, 'The style missing could not be loaded: HTTP 404');
  });
  assert.deepEqual([...missing], [], 'every text these tests asked for exists');
});
