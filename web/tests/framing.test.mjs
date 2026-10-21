// Run from the repo root: node --test 'web/tests/*.test.mjs'
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { fitDistance, fitView, framingPoints, photoSideBearing, viewDirection } from '../framing.js';

const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

// Projects p for a camera at target + dir * distance looking at target with y up: [x, y] in tangent
// units (in view when within tan(half fov)), plus the depth in front of the eye.
function project(p, target, dir, distance) {
  const z = dir;
  const xl = Math.hypot(z[2], z[0]);
  const x = [z[2] / xl, 0, -z[0] / xl];
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const q = p.map((v, k) => v - target[k]);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const depth = distance - dot(q, z);
  return { sx: dot(q, x) / depth, sy: dot(q, y) / depth, depth };
}

function inView(points, target, dir, distance, fovDeg, aspect) {
  const tanV = Math.tan((fovDeg * Math.PI) / 360);
  return points.every((p) => {
    const { sx, sy, depth } = project(p, target, dir, distance);
    return depth > 0 && Math.abs(sy) <= tanV + 1e-9 && Math.abs(sx) <= tanV * aspect + 1e-9;
  });
}

// A Baptistry-like facade seen from the east: cloud box ~30 m across around x = 45, photographers 25-45 m
// further east at eye height (three.js axes: x east, y up, z south).
const CLOUD = [];
for (const x of [38, 52]) for (const y of [0, 26]) for (const z of [-15, 15]) CLOUD.push([x, y, z]);
const CAMERAS = [
  [75, 1.6, -10],
  [80, 1.6, 5],
  [90, 1.6, 20],
  [70, 1.6, 12],
];

test('viewDirection: bearing clockwise from north, north is -z, elevation lifts y', () => {
  const [ex, ey, ez] = viewDirection(90, 0);
  assert.ok(near(ex, 1) && near(ey, 0) && near(ez, 0));
  const [nx, , nz] = viewDirection(0, 0);
  assert.ok(near(nx, 0) && near(nz, -1));
  const up = viewDirection(140, 30);
  assert.ok(near(Math.hypot(...up), 1) && near(up[1], 0.5));
});

test('fitDistance puts every point in view, for wide and narrow viewers', () => {
  const points = [...CLOUD, ...CAMERAS];
  const dir = viewDirection(90, 30);
  for (const aspect of [758 / 520, 4 / 3, 0.6]) {
    const target = [60, 10, 0];
    const d = fitDistance(points, target, dir, 50, aspect);
    assert.ok(inView(points, target, dir, d, 50, aspect), `aspect ${aspect}`);
    assert.ok(!inView(points, target, dir, d * 0.8, 50, aspect), 'and not much farther than needed');
  }
});

test('fitView centres the points on screen, so the model is not shrunk by empty margins', () => {
  const points = [...CLOUD, ...CAMERAS];
  const dir = viewDirection(90, 30);
  const { target, distance } = fitView(points, dir, 50, 4 / 3);
  assert.ok(inView(points, target, dir, distance, 50, 4 / 3));
  const s = points.map((p) => project(p, target, dir, distance));
  const mid = (v) => (Math.min(...v) + Math.max(...v)) / 2;
  assert.ok(Math.abs(mid(s.map((p) => p.sx))) < 0.02 && Math.abs(mid(s.map((p) => p.sy))) < 0.02);
  // Tighter than fitting the same points around their bounding-box centre.
  const lo = [0, 1, 2].map((k) => Math.min(...points.map((p) => p[k])));
  const hi = [0, 1, 2].map((k) => Math.max(...points.map((p) => p[k])));
  assert.ok(distance < fitDistance(points, lo.map((v, k) => (v + hi[k]) / 2), dir, 50, 4 / 3));
});

test('framingPoints leaves out cameras placed far from the model', () => {
  const centre = [45, 13, 0];
  const radius = 15;
  const far = [400, 1.6, 0];
  assert.deepEqual(framingPoints([...CAMERAS, far], centre, radius), CAMERAS);
  assert.deepEqual(framingPoints([], centre, radius), []);
});

test('photoSideBearing: look from the side the photos cluster on, else the default view', () => {
  const at = (deg, r = 60) => [r * Math.sin((deg * Math.PI) / 180), r * Math.cos((deg * Math.PI) / 180), 1.6];
  // Baptistry-like: photographers east, north-east and south-east of the landmark.
  const east = [at(80), at(90), at(95), at(100), at(60), at(45), at(130), at(140)];
  assert.ok(Math.abs(photoSideBearing(east) - 92) < 5, String(photoSideBearing(east)));
  // Bearings around north average across 0/360.
  assert.ok(Math.abs(photoSideBearing([at(350), at(10), at(355)]) - 358.3) < 0.5);
  // All around, or six of eight sectors as the default full-model rule asks: no single side.
  assert.equal(photoSideBearing([0, 45, 90, 135, 180, 225, 270, 315].map((d) => at(d))), null);
  assert.equal(photoSideBearing([0, 45, 90, 135, 180, 225].map((d) => at(d))), null);
  assert.equal(photoSideBearing([]), null);
  assert.equal(photoSideBearing([[0, 0, 1.6]]), null, 'a camera on the centre has no bearing');
});
