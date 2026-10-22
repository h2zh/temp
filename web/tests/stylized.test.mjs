// Run from the repo root: node --test 'web/tests/*.test.mjs'
// The pure helpers of the stylized-model viewer (three.js is only imported inside main(), in a browser).
import assert from 'node:assert/strict';
import { test } from 'node:test';

globalThis.__STYLIZED_NO_AUTOSTART = true;
const {
  TOON_STEPS,
  bearingOf,
  describe,
  insetFrame,
  parseViewParams,
  pickLockedTags,
  preferredBearing,
  summarize,
  toonGradientData,
  viewDirection,
} = await import('../stylized.js');

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
