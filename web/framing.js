// Where the 3D viewer puts its camera. Plain numbers in three.js world coordinates (x = east, y = up,
// z = south), so node tests cover it without WebGL.

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const add = (a, b, k = 1) => [a[0] + k * b[0], a[1] + k * b[1], a[2] + k * b[2]];

function normalize(v) {
  const n = Math.hypot(...v);
  return v.map((c) => c / n);
}

/** Unit vector from the target towards the eye for a compass bearing (clockwise from north) and elevation. */
export function viewDirection(bearingDeg, elevationDeg) {
  const a = (bearingDeg * Math.PI) / 180;
  const e = (elevationDeg * Math.PI) / 180;
  return [Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e)];
}

/**
 * Photo positions worth zooming out for: those within `reach` cloud radii of the cloud centre. A camera
 * that registration put far off would otherwise shrink the model to a dot.
 */
export function framingPoints(points, centre, radius, reach = 5) {
  return points.filter((p) => Math.hypot(...sub(p, centre)) <= reach * radius);
}

/**
 * Eye distance from `target` along the unit vector `dir` at which every point is in view of a perspective
 * camera (vertical fov `fovDeg`, `aspect` = width / height) that looks at the target with y up.
 * `margin` 1.1 leaves 10 % spare on the tightest side.
 */
export function fitDistance(points, target, dir, fovDeg, aspect, margin = 1.1) {
  const z = normalize(dir); // the camera looks along -z
  const x = normalize(cross([0, 1, 0], z));
  const y = cross(z, x);
  const tanV = Math.tan((fovDeg * Math.PI) / 360);
  const tanH = tanV * aspect;
  let distance = 0;
  for (const p of points) {
    const q = sub(p, target);
    const along = dot(q, z);
    // In view when |lateral| <= (distance - along) * tan(half fov); the +1 keeps every point 1 m past the eye.
    distance = Math.max(
      distance,
      along + (margin * Math.abs(dot(q, x))) / tanH,
      along + (margin * Math.abs(dot(q, y))) / tanV,
      along + 1,
    );
  }
  return distance;
}

/**
 * Target and eye distance that fit every point into the view from `dir` (see fitDistance) with the points
 * centred on screen, so no side of the view is left empty.
 */
export function fitView(points, dir, fovDeg, aspect, margin = 1.1) {
  const z = normalize(dir);
  const x = normalize(cross([0, 1, 0], z));
  const y = cross(z, x);
  const lo = [0, 1, 2].map((k) => Math.min(...points.map((p) => p[k])));
  const hi = [0, 1, 2].map((k) => Math.max(...points.map((p) => p[k])));
  let target = lo.map((v, k) => (v + hi[k]) / 2);
  let distance = fitDistance(points, target, z, fovDeg, aspect, margin);
  for (let i = 0; i < 4; i++) {
    // Screen offsets as tangents of the view angle; recentre on the middle of their range.
    const sx = [];
    const sy = [];
    for (const p of points) {
      const q = sub(p, target);
      const depth = distance - dot(q, z);
      sx.push(dot(q, x) / depth);
      sy.push(dot(q, y) / depth);
    }
    const mid = (v) => (Math.min(...v) + Math.max(...v)) / 2;
    target = add(add(target, x, mid(sx) * distance), y, mid(sy) * distance);
    distance = fitDistance(points, target, z, fovDeg, aspect, margin);
  }
  return { target, distance };
}

/**
 * Compass bearing to first look at a full model from: the side its photos were taken from, when they
 * cluster on one side (mean resultant length of their bearings >= minConcentration), else null for the
 * viewer's default. Real landmarks are often photographed from one or two streets only, and the far side
 * of such a model is barely reconstructed. positions: ENU [east, north, ...] from the landmark centre.
 */
export function photoSideBearing(positions, minConcentration = 0.5) {
  let east = 0;
  let north = 0;
  let n = 0;
  for (const [e, nn] of positions) {
    const r = Math.hypot(e, nn);
    if (!r) continue;
    east += e / r;
    north += nn / r;
    n++;
  }
  if (!n || Math.hypot(east, north) / n < minConcentration) return null;
  return ((Math.atan2(east, north) * 180) / Math.PI + 360) % 360;
}
