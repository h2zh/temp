// Run from the repo root: node --test 'web/tests/*.test.mjs'
// The pure helpers of the stylized-model viewer (three.js is only imported inside main(), in a browser).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

globalThis.__STYLIZED_NO_AUTOSTART = true;
const {
  AUTO_ADOPTED_LABEL,
  AUTO_SOURCE,
  GEOMETRY_TEXT,
  MAX_FRAMES,
  MIN_FRAME_AREA,
  PROVENANCE_TEXT,
  TOON_STEPS,
  bearingOf,
  describe,
  insetFrame,
  parseViewParams,
  pickFrames,
  pickLockedTags,
  pickLook,
  pickerChoices,
  preferredBearing,
  summarize,
  themeSearch,
  toonGradientData,
  viewDirection,
} = await import('../stylized.js');

const { DEFAULT_THEME, hexToRgb, rgbKey } = await import('../themes.js');
const TABLE = JSON.parse(readFileSync(new URL('../themes/palette.json', import.meta.url), 'utf8')).colors;

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
  const [x, y, z] = viewDirection(90, 0);
  near(x, 1);
  near(y, 0);
  near(z, 0, 1e-12);
  near(Math.hypot(...viewDirection(33, 27)), 1);
});

test('the first view faces the photographed walls, weighted by area', () => {
  const walls = [
    { normal: [0, 0, 1], area: 100, observed: true }, // facing south
    { normal: [1, 0, 0], area: 20, observed: true }, // east, smaller
    { normal: [0, 0, -1], area: 500, observed: false }, // north, locked: ignored
  ];
  const b = preferredBearing(walls);
  assert.ok(b > 160 && b < 180, String(b));
  assert.equal(preferredBearing([{ normal: [0, 0, 1], area: 5, observed: false }], 140), 140);
});

test('panel counts and the provenance shown for a tapped face', () => {
  const extras = [
    // wall 0: a photographed strip and a clay strip beside it (one wall)
    { kind: 'wall', wall: 0, observed: true, locked: false, observed_fraction: 0.8, photos: [{ photo: 'a', share: 0.6 }, { photo: 'b', share: 0.4 }],
      provenance: { outline: 'osm-footprint', height: 'sfm-points', appearance: 'photo-observed' } },
    { kind: 'wall', wall: 0, observed: false, locked: true, provenance: { outline: 'osm-footprint', height: 'sfm-points', appearance: 'default-style' } },
    { kind: 'wall', wall: 1, observed: false, locked: true, provenance: { outline: 'osm-footprint', height: 'inferred', appearance: 'default-style' } },
    // a pyramid: faces 0-2 observed as a whole (60 m2), faces 3-5 clay (40 m2) with a 10 m2 observed patch
    { kind: 'roof', faces: [0, 1, 2], area_m2: 60, observed: true, locked: false, coverage: [0.9, 1, 0.85], provenance: { outline: 'osm-footprint', height: 'sfm-points', appearance: 'photo-observed' } },
    { kind: 'roof', faces: [3, 4, 5], area_m2: 40, observed: false, locked: true, coverage: [0.3, 0, 0], provenance: { outline: 'osm-footprint', height: 'default-style', appearance: 'default-style' } },
    { kind: 'roof-patch', faces: [3], area_m2: 10, observed: true, locked: false, coverage: [0.7], provenance: { outline: 'osm-footprint', height: 'default-style', appearance: 'photo-observed' } },
    { kind: 'corners', observed: false, locked: true },
  ];
  const s = summarize(extras);
  assert.deepEqual(
    { ...s, roofObservedShare: Math.round(s.roofObservedShare * 100) },
    { walls: 2, observedWalls: 1, lockedWalls: 1, roofFaces: 6, observedRoofFaces: 3, roofObservedShare: 70, withoutProvenance: 1 },
  );
  const lines = describe(extras[2]);
  assert.match(lines[0], /立面 #1 · 待解锁/);
  assert.match(lines[2], /推断/);
  assert.match(lines[3], /默认样式/);
  const seen = describe(extras[0]);
  assert.match(seen[3], /照片实拍/);
  assert.doesNotMatch(seen[2], /取自/);
  assert.ok(seen.some((l) => /实拍覆盖：80%/.test(l)) && seen.some((l) => /2 张照片/.test(l)));
  assert.ok(describe(extras[4]).some((l) => /点云覆盖：30% \/ 0% \/ 0%/.test(l)));
  assert.match(describe(extras[5])[0], /屋顶实拍区/);
  // D12: small enclosed holes filled with the wall's base colour are named as such, apart from the coverage
  const filled = describe({ ...extras[0], fill: { holes: 2, area_m2: 0.4321, px_per_m: 16 } });
  assert.ok(filled.some((l) => /底色补洞：2 处，共 0.43 m²/.test(l) && /不算实拍覆盖/.test(l)), filled.join('\n'));
  assert.ok(!describe({ ...extras[0], fill: { holes: 0, area_m2: 0 } }).some((l) => /底色补洞/.test(l)));
  // models written before 'inferred' existed
  assert.match(describe({ ...extras[1], height_measured_here: false })[2], /重建点云实测（取自同一体块已测的墙）/);
  assert.deepEqual(describe({}), []);
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

test('locked tags go on the biggest locked walls, spread apart', () => {
  const w = (area, x, locked = true) => ({ area, locked, center: [x, 0, 0] });
  const tags = pickLockedTags([w(10, 0), w(50, 1), w(40, 20), w(30, 40), w(99, 5, false)], 10, 2);
  assert.deepEqual(tags.map((t) => t.area), [50, 40]);
});

test('dense parts models frame only their largest locked walls', () => {
  const walls = Array.from({ length: 300 }, (_, i) => ({ area: i + 1, locked: i % 7 !== 0 }));
  const f = pickFrames(walls);
  assert.equal(f.length, MAX_FRAMES);
  assert.ok(f.every((w) => w.locked && w.area >= MIN_FRAME_AREA));
  assert.deepEqual(f.slice(0, 2).map((w) => w.area), [300, 299]);
  assert.deepEqual(pickFrames([{ area: 5, locked: true }, { area: 20, locked: false }]), []);
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
  assert.ok(lines.some((l) => /待解锁/.test(l)), 'appearance stays clay: the adoption is geometry only');
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

test('the dashed frame of a locked wall sits inside the wall, just off its face', () => {
  const corners = [[0, 0, 0], [10, 0, 0], [10, 6, 0], [0, 6, 0]]; // a 10 m x 6 m wall facing +z
  const seg = insetFrame(corners, [0, 0, 1], 0.8, 0.05);
  assert.equal(seg.length, 24);
  const pts = [];
  for (let i = 0; i < 24; i += 3) pts.push(seg.slice(i, i + 3));
  for (const [x, y, z] of pts) {
    assert.ok(x >= 0.72 - 1e-9 && x <= 10 - 0.72 + 1e-9 && y >= 0.72 - 1e-9 && y <= 6 - 0.72 + 1e-9, `${x},${y}`);
    near(z, 0.05);
  }
  near(pts[0][0], 0.72); // inset capped at 12 % of the 6 m side
});

test('the default theme also remaps the atlases: their clay becomes its clay', () => {
  const notes = [];
  const { look, colors, restyle, check } = pickLook(DEFAULT_THEME, TABLE, 'default', undefined, notes);
  assert.ok(check.ok && restyle && notes.length === 0);
  assert.equal(look.id, 'default');
  const clay = TABLE.find((r) => r.role === 'clay').rgb;
  assert.equal(colors.map.get(rgbKey(clay)), rgbKey(hexToRgb(DEFAULT_THEME.clay.wall)));
  assert.notEqual(rgbKey(clay), rgbKey(hexToRgb(DEFAULT_THEME.clay.wall)));
});

test('a theme that fails the validator gives way to the default, with a note', () => {
  const bad = structuredClone(DEFAULT_THEME);
  bad.id = 'bad';
  bad.name = '坏画风';
  bad.clay = { ...bad.clay, wall: bad.ink, hatch: bad.ink };
  const notes = [];
  const { look, check } = pickLook(bad, TABLE, 'bad', undefined, notes);
  assert.equal(look.id, 'default');
  assert.ok(check.ok);
  assert.equal(notes.length, 1);
  assert.match(notes[0], /坏画风.*没有通过校验/);
});

test('without a colour table nothing is remapped', () => {
  const { colors, restyle } = pickLook(DEFAULT_THEME, null, 'default', undefined, []);
  assert.equal(colors, null);
  assert.equal(restyle, false);
});
