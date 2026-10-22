// Viewer for the stylized ("toy" / anime) landmark models written by crowd3d/stylize.py.
//
// The GLB is already y-up (ENU x, y, z -> three x, z, -y; north is -Z). Every node carries its
// provenance in extras (three.js puts them in userData): kind (wall / corners / roof / roof-patch /
// roof-cut), observed, locked, provenance {outline, height, appearance}, plus area_m2, coverage /
// observed_fraction and the contributing photos of observed walls. A wall can be several nodes (its
// photographed strip and clay strips beside it). The look: MeshToonMaterial with a 3-step
// gradient, an inverted-hull silhouette of constant pixel width, thick crease lines, a soft sky and
// a ground disc. Faces nobody photographed ("locked") are cool light clay with a faint hatch and carry a
// "待解锁" tag: they show the real footprint and height only, never invented detail.
//
// Query parameters: model=<GLB url> (required), bearing=<deg, compass bearing from the building to
// the camera; default: the side with the most photographed facades>, elev=<deg>, zoom=<factor>,
// ui=0 (hide the panel, for screenshots), outline=<px>.
//
// The pure helpers at the top are exported for node tests; three.js is imported lazily so that
// importing this module outside a browser does not need it.

export const TOON_STEPS = [0.5, 0.78, 1.0];
export const INK = 0x2d2a32;
export const CLAY = 0xe4e8ee; // = crowd3d.stylize.CLAY_RGB
export const LOCKED_LABEL = '待解锁';
export const PROVENANCE_TEXT = {
  'photo-observed': '照片实拍',
  'osm-footprint': 'OSM 轮廓',
  'sfm-points': '重建点云实测',
  'osm-height': 'OSM 高度标签',
  'wikidata-height': 'Wikidata 高度',
  inferred: '推断（同一体块别处实测，这里没测到）',
  'default-style': '默认样式（未拍到）',
};

/** RGBA bytes of an n-step toon ramp (three.js reads the red channel). */
export function toonGradientData(steps = TOON_STEPS) {
  const data = new Uint8Array(steps.length * 4);
  steps.forEach((v, i) => {
    const b = Math.round(Math.max(0, Math.min(1, v)) * 255);
    data.set([b, b, b, 255], i * 4);
  });
  return data;
}

function num(params, key, fallback, lo, hi) {
  const raw = params.get(key);
  if (raw === null || raw.trim() === '') return fallback;
  const v = Number(raw);
  return Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : fallback;
}

/** Viewer options from a location.search string. Only same-origin or relative model URLs are accepted. */
export function parseViewParams(search, origin = 'http://localhost') {
  const params = new URLSearchParams(search);
  let model = params.get('model');
  if (model) {
    try {
      const url = new URL(model, origin + '/');
      model = url.origin === new URL(origin).origin ? url.pathname + url.search : null;
    } catch {
      model = null;
    }
  }
  return {
    model: model || null,
    bearing: params.has('bearing') ? num(params, 'bearing', null, -360, 720) : null,
    elevation: num(params, 'elev', 24, 2, 85),
    zoom: num(params, 'zoom', 1, 0.3, 4),
    outlinePx: num(params, 'outline', 2.5, 0, 8),
    ui: params.get('ui') !== '0',
  };
}

/** Compass bearing (deg, 0 = north, 90 = east) of a three.js direction (x east, z south). */
export function bearingOf(x, z) {
  return ((Math.atan2(x, -z) * 180) / Math.PI + 360) % 360;
}

/**
 * Default viewpoint: the area-weighted mean outward normal of the observed walls, so the first view
 * shows what people actually photographed. walls: [{normal: [x, y, z], area, observed}].
 */
export function preferredBearing(walls, fallback = 140) {
  let x = 0;
  let z = 0;
  for (const w of walls) {
    if (!w.observed) continue;
    x += w.normal[0] * w.area;
    z += w.normal[2] * w.area;
  }
  return Math.hypot(x, z) < 1e-6 ? fallback : bearingOf(x, z);
}

/**
 * Counts for the panel from the nodes' extras. Walls are counted once each (a wall may be split into a
 * photographed strip and clay strips); roof faces are counted from extras.faces, and the roof's observed
 * share is by area (whole faces observed plus observed patches).
 */
export function summarize(extrasList) {
  const list = extrasList.filter((e) => e && e.kind);
  const wallParts = list.filter((e) => e.kind === 'wall');
  const wallIds = new Set(wallParts.map((e, i) => e.wall ?? `part-${i}`));
  const observedIds = new Set(wallParts.filter((e) => e.observed).map((e, i) => e.wall ?? `part-${i}`));
  const roofs = list.filter((e) => e.kind === 'roof');
  const nFaces = (e) => (Array.isArray(e.faces) ? e.faces.length : 1);
  const area = (e) => (Number.isFinite(e.area_m2) ? e.area_m2 : 0);
  const roofArea = roofs.reduce((s, e) => s + area(e), 0);
  const observedRoofArea =
    roofs.filter((e) => e.observed).reduce((s, e) => s + area(e), 0) +
    list.filter((e) => e.kind === 'roof-patch' && e.observed).reduce((s, e) => s + area(e), 0);
  return {
    walls: wallIds.size,
    observedWalls: observedIds.size,
    lockedWalls: wallIds.size - observedIds.size,
    roofFaces: roofs.reduce((s, e) => s + nFaces(e), 0),
    observedRoofFaces: roofs.filter((e) => e.observed).reduce((s, e) => s + nFaces(e), 0),
    roofObservedShare: roofArea > 0 ? Math.min(1, observedRoofArea / roofArea) : 0,
    withoutProvenance: list.filter((e) => !e.provenance).length,
  };
}

/** Lines describing a picked face. */
export function describe(extras) {
  if (!extras || !extras.kind) return [];
  const kind = { wall: '立面', roof: '屋顶', 'roof-patch': '屋顶实拍区', 'roof-cut': '屋顶切面', corners: '转角' }[extras.kind] ?? extras.kind;
  const p = extras.provenance ?? {};
  const t = (v) => PROVENANCE_TEXT[v] ?? v ?? '未知';
  const pct = (v) => `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%`;
  const lines = [
    `${kind}${extras.wall !== undefined ? ' #' + extras.wall : ''} · ${extras.locked ? LOCKED_LABEL : '已解锁'}`,
    `轮廓：${t(p.outline)}`,
    // older models said sfm-points with height_measured_here=false for carried-over heights
    `高度：${t(p.height)}${extras.height_measured_here === false && p.height === 'sfm-points' ? '（取自同一体块已测的墙）' : ''}`,
    `外观：${t(p.appearance)}`,
  ];
  if (Number.isFinite(extras.observed_fraction)) lines.push(`实拍覆盖：${pct(extras.observed_fraction)}（其余为素模）`);
  if (Array.isArray(extras.coverage) && extras.coverage.length && extras.kind !== 'wall') {
    lines.push(`点云覆盖：${extras.coverage.map(pct).join(' / ')}`);
  }
  if (Array.isArray(extras.photos) && extras.photos.length) lines.push(`来自 ${extras.photos.length} 张照片`);
  return lines;
}

/** Camera offset (three.js axes) for a compass bearing and elevation, unit length. */
export function viewDirection(bearingDeg, elevationDeg) {
  const b = (bearingDeg * Math.PI) / 180;
  const e = (elevationDeg * Math.PI) / 180;
  return [Math.sin(b) * Math.cos(e), Math.sin(e), -Math.cos(b) * Math.cos(e)];
}

/** Up to `max` locked walls to tag, largest first, at least minGap apart (centres in three.js units). */
export function pickLockedTags(walls, minGap, max = 3) {
  const out = [];
  for (const w of [...walls].filter((w) => w.locked).sort((a, b) => b.area - a.area)) {
    if (out.length >= max) break;
    if (out.every((o) => Math.hypot(o.center[0] - w.center[0], o.center[2] - w.center[2]) >= minGap)) out.push(w);
  }
  return out;
}

/**
 * Dashed "placeholder" frame inset into a locked wall quad: corners [bottom-left, bottom-right,
 * top-right, top-left] (three.js axes), the wall's outward normal; returns 4 segments as a flat
 * xyz array (24 numbers), inset by `inset` (capped at 12 % of the smaller side) and lifted `lift`
 * off the wall along the normal.
 */
export function insetFrame(corners, normal, inset = 0.8, lift = 0.05) {
  const [a, b, c, d] = corners;
  const sub = (p, q) => p.map((v, i) => v - q[i]);
  const len = (v) => Math.hypot(...v);
  const width = len(sub(b, a));
  const height = len(sub(d, a));
  const k = Math.min(inset, 0.12 * Math.min(width, height));
  const u = sub(b, a).map((v) => v / (width || 1));
  const w = sub(d, a).map((v) => v / (height || 1));
  const at = (p, su, sw) => p.map((v, i) => v + su * u[i] + sw * w[i] + lift * normal[i]);
  const q = [at(a, k, k), at(b, -k, k), at(c, -k, -k), at(d, k, -k)];
  return [0, 1, 2, 3].flatMap((i) => [...q[i], ...q[(i + 1) % 4]]);
}

// ------------------------------------------------------------------ browser part

const HATCH_VERTEX = [
  '#include <common>',
  'varying vec3 vHatchPos;',
].join('\n');
const HATCH_FRAGMENT = [
  '#include <common>',
  'varying vec3 vHatchPos;',
].join('\n');

function addHatch(material) {
  // faint diagonal stripes in world space on locked faces ("not unlocked yet")
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', HATCH_VERTEX)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvHatchPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', HATCH_FRAGMENT)
      .replace(
        '#include <color_fragment>',
        '#include <color_fragment>\nfloat hatch = fract((vHatchPos.x + vHatchPos.y - vHatchPos.z) / 1.8);\n' +
          'diffuseColor.rgb *= mix(1.0, 0.84, step(0.72, hatch));',
      );
  };
  material.customProgramCacheKey = () => 'crowd3d-hatch';
}

function outlineMaterial(THREE, px, resolution) {
  // inverted hull, offset in clip space so the silhouette is `px` pixels wide at any distance
  return new THREE.ShaderMaterial({
    uniforms: { thickness: { value: px }, resolution: { value: resolution }, ink: { value: new THREE.Color(INK) } },
    vertexShader: `
      uniform float thickness;
      uniform vec2 resolution;
      void main() {
        vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        vec3 n = normalize(normalMatrix * normal);
        vec2 dir = (projectionMatrix * vec4(n, 0.0)).xy;
        float len = length(dir);
        if (len > 1e-6) clip.xy += dir / len * thickness * 2.0 / resolution * clip.w;
        gl_Position = clip;
      }`,
    fragmentShader: `
      uniform vec3 ink;
      void main() { gl_FragColor = vec4(ink, 1.0); }`,
    side: THREE.BackSide,
  });
}

async function main() {
  const THREE = await import('three');
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
  const { mergeGeometries, mergeVertices } = await import('three/addons/utils/BufferGeometryUtils.js');
  const { LineSegments2 } = await import('three/addons/lines/LineSegments2.js');
  const { LineSegmentsGeometry } = await import('three/addons/lines/LineSegmentsGeometry.js');
  const { LineMaterial } = await import('three/addons/lines/LineMaterial.js');
  const { CSS2DRenderer, CSS2DObject } = await import('three/addons/renderers/CSS2DRenderer.js');

  const opts = parseViewParams(location.search, location.origin);
  const stage = document.getElementById('stage');
  const panel = document.getElementById('panel');
  const status = document.getElementById('status');
  const pick = document.getElementById('pick');
  if (!opts.ui) document.body.classList.add('no-ui');
  const state = (window.__stylized = { ready: false, error: null, stats: null });

  const fail = (msg) => {
    status.textContent = msg;
    status.hidden = false;
    state.error = msg;
    state.ready = true;
  };
  if (!opts.model) return fail('缺少模型：在地址后加 ?model=模型.glb');

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0); // the CSS sky shows through
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  stage.append(renderer.domElement);
  const labels = new CSS2DRenderer();
  labels.domElement.className = 'labels';
  stage.append(labels.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.5, 5000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.maxPolarAngle = Math.PI * 0.48;

  // intensities in three.js physical units: albedo x (hemisphere ~0.4 + sun ~0.65 x toon step)
  scene.add(new THREE.HemisphereLight(0xfdfbff, 0xd5e6c4, 1.25));
  const sun = new THREE.DirectionalLight(0xfffaf0, 2.0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0015;
  sun.shadow.normalBias = 0.08; // no shadow acne stripes on large flat walls
  scene.add(sun, sun.target);

  const gradient = new THREE.DataTexture(toonGradientData(), TOON_STEPS.length, 1, THREE.RGBAFormat);
  gradient.minFilter = gradient.magFilter = THREE.NearestFilter;
  gradient.needsUpdate = true;

  let gltf;
  try {
    gltf = await new GLTFLoader().loadAsync(opts.model);
  } catch (e) {
    return fail(`模型加载失败：${e.message ?? e}`);
  }
  const model = gltf.scene;
  const info = model.userData ?? {};
  model.updateMatrixWorld(true);

  const meshes = [];
  const walls = [];
  const toon = new Map(); // one toon material per source material (atlases and flat colours are shared)
  model.traverse((o) => {
    if (!o.isMesh) return;
    const ex = o.userData ?? {};
    const old = o.material;
    const key = `${old.uuid}|${ex.locked && !old.map ? ex.kind : ''}`;
    let mat = toon.get(key);
    if (!mat) {
      mat = new THREE.MeshToonMaterial({
        color: old.map ? 0xffffff : old.color,
        map: old.map ?? null,
        gradientMap: gradient,
      });
      if (ex.locked && !old.map) {
        mat.color = new THREE.Color(CLAY).convertSRGBToLinear();
        if (ex.kind === 'roof') mat.color.multiplyScalar(0.93);
        addHatch(mat);
      }
      toon.set(key, mat);
    }
    o.material = mat;
    meshes.push(o);
    if (ex.kind === 'wall') {
      o.geometry.computeBoundingBox();
      const box = o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld);
      const size = box.getSize(new THREE.Vector3());
      const n = o.geometry.getAttribute('normal');
      const pos = o.geometry.getAttribute('position');
      const corners = [0, 1, 2, 3].map((i) => new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld).toArray());
      walls.push({
        corners,
        normal: [n.getX(0), n.getY(0), n.getZ(0)],
        area: Math.hypot(size.x, size.z) * size.y,
        observed: !!ex.observed,
        locked: !!ex.locked,
        center: box.getCenter(new THREE.Vector3()).toArray(),
        top: box.max.y,
      });
    }
  });
  // draw: one merged mesh per material (a few draw calls instead of one per face group); each keeps
  // the triangle ranges of its face groups so a tap still finds the face's own provenance
  const drawn = [];
  const byMaterial = new Map();
  for (const m of meshes) {
    if (!byMaterial.has(m.material)) byMaterial.set(m.material, []);
    byMaterial.get(m.material).push(m);
  }
  for (const [mat, list] of byMaterial) {
    const geos = list.map((m) => {
      const g = m.geometry.index ? m.geometry.toNonIndexed() : m.geometry.clone();
      for (const name of Object.keys(g.attributes)) if (!['position', 'normal', 'uv'].includes(name)) g.deleteAttribute(name);
      return g.applyMatrix4(m.matrixWorld);
    });
    const ranges = [];
    let start = 0;
    geos.forEach((g, i) => {
      const n = g.getAttribute('position').count / 3;
      ranges.push([start, start + n, list[i].userData]);
      start += n;
    });
    const mesh = new THREE.Mesh(mergeGeometries(geos), mat);
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.userData.ranges = ranges;
    drawn.push(mesh);
    scene.add(mesh);
  }
  const extrasAt = (hit) => hit.object.userData.ranges.find(([a, b]) => hit.faceIndex >= a && hit.faceIndex < b)?.[2];

  // silhouette + crease lines from one welded copy of the whole model
  const welded = mergeVertices(
    mergeGeometries(
      meshes.map((m) => {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', m.geometry.getAttribute('position').clone());
        if (m.geometry.index) g.setIndex(m.geometry.index.clone());
        g.applyMatrix4(m.matrixWorld);
        return g.index ? g.toNonIndexed() : g;
      }),
    ),
    1e-3,
  );
  welded.computeVertexNormals();
  const resolution = new THREE.Vector2(1, 1);
  if (opts.outlinePx > 0) {
    scene.add(new THREE.Mesh(welded, outlineMaterial(THREE, opts.outlinePx, resolution)));
  }
  const creases = new LineSegmentsGeometry().fromEdgesGeometry(new THREE.EdgesGeometry(welded, 35));
  const lineMat = new LineMaterial({ color: INK, linewidth: Math.max(1, opts.outlinePx * 0.7), resolution });
  scene.add(new LineSegments2(creases, lineMat));
  // locked walls: a dashed placeholder frame ("to be unlocked")
  const frames = walls.filter((w) => w.locked).flatMap((w) => insetFrame(w.corners, w.normal));
  let dashMat = null;
  if (frames.length) {
    const g = new LineSegmentsGeometry().setPositions(frames);
    dashMat = new LineMaterial({ color: 0x8f9aab, linewidth: Math.max(1, opts.outlinePx * 0.8), dashed: true,
      dashSize: 0.9, gapSize: 0.6, resolution });
    const dashes = new LineSegments2(g, dashMat);
    dashes.computeLineDistances();
    scene.add(dashes);
  }

  // ground disc and framing
  const box = new THREE.Box3().setFromObject(model); // (not added to the scene: `drawn` is)
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(4, 0.5 * Math.hypot(size.x, size.y, size.z));
  const footR = 0.5 * Math.hypot(size.x, size.z);
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(footR * 1.3, 96),
    new THREE.MeshToonMaterial({ color: 0xcfeab8, gradientMap: gradient }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(center.x, -0.02, center.z);
  ground.receiveShadow = true;
  const rim = new THREE.Mesh(
    new THREE.RingGeometry(footR * 1.3, footR * 1.37, 96),
    new THREE.MeshBasicMaterial({ color: 0xa9d48c }),
  );
  rim.rotation.x = -Math.PI / 2;
  rim.position.set(center.x, -0.01, center.z);
  scene.add(ground, rim);

  const s = 1.8 * footR + size.y;
  sun.position.set(center.x - 0.5 * s, center.y + 1.2 * s, center.z + 0.8 * s);
  sun.target.position.copy(center);
  Object.assign(sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: 0.1, far: 4 * s });
  sun.shadow.camera.updateProjectionMatrix();

  const bearing = opts.bearing ?? preferredBearing(walls);
  const target = new THREE.Vector3(center.x, box.min.y + 0.45 * size.y, center.z);
  const dir = viewDirection(bearing, opts.elevation);
  let dist = radius / Math.sin((camera.fov * Math.PI) / 360);
  function place() {
    camera.position.set(target.x + dir[0] * dist, target.y + dir[1] * dist, target.z + dir[2] * dist);
    camera.near = dist / 50;
    camera.far = dist * 20;
    controls.target.copy(target);
    controls.maxDistance = dist * 3;
    controls.minDistance = radius * 0.4;
    controls.update();
    camera.updateMatrixWorld();
  }
  function frame() {
    // fit the model's box to ~80 % of the view (refined twice: perspective is not linear in distance)
    for (let i = 0; i < 3; i++) {
      place();
      camera.updateProjectionMatrix();
      let ext = 0;
      for (const x of [box.min.x, box.max.x])
        for (const y of [box.min.y, box.max.y])
          for (const z of [box.min.z, box.max.z]) {
            const v = new THREE.Vector3(x, y, z).project(camera);
            ext = Math.max(ext, Math.abs(v.x), Math.abs(v.y));
          }
      dist *= Math.max(0.5, Math.min(2, ext / 0.8));
    }
    dist /= opts.zoom;
    place();
  }

  // "待解锁" tags on the biggest locked walls
  const tags = [];
  for (const w of pickLockedTags(walls, radius * 0.6)) {
    const el = document.createElement('div');
    el.className = 'lock-tag';
    el.textContent = `🔒 ${LOCKED_LABEL}`;
    const tag = new CSS2DObject(el);
    // on the wall's face, a little above its middle; shown only while the wall faces the camera
    tag.position.set(w.center[0] + w.normal[0] * 0.6, w.center[1] + 0.15 * (w.top - w.center[1]), w.center[2] + w.normal[2] * 0.6);
    tag.userData.normal = new THREE.Vector3(...w.normal);
    tags.push(tag);
    scene.add(tag);
  }
  const toCamera = new THREE.Vector3();
  const tagRay = new THREE.Raycaster();
  function updateTags() {
    // shown while the wall faces the camera and nothing (another block) stands between them
    for (const tag of tags) {
      toCamera.subVectors(camera.position, tag.position);
      const dist = toCamera.length();
      toCamera.normalize();
      let visible = toCamera.dot(tag.userData.normal) > 0.15;
      if (visible) {
        tagRay.set(camera.position, toCamera.clone().negate());
        tagRay.far = dist - 0.3;
        visible = tagRay.intersectObjects(drawn, false).length === 0;
      }
      tag.visible = visible;
    }
  }

  // panel
  const stats = summarize(meshes.map((m) => m.userData));
  state.stats = { ...stats, bearing: Math.round(bearing), triangles: welded.index ? welded.index.count / 3 : 0 };
  document.getElementById('title').textContent = info.landmark ?? '地标';
  document.getElementById('unlocked').textContent = `${stats.observedWalls}/${stats.walls}`;
  document.getElementById('roofs').textContent = `${Math.round(stats.roofObservedShare * 100)}%`;
  const attribution = document.getElementById('attribution');
  for (const line of info.attribution ?? []) {
    const p = document.createElement('p');
    p.textContent = line;
    attribution.append(p);
  }
  panel.hidden = !opts.ui;

  // tap a face -> its provenance
  const ray = new THREE.Raycaster();
  let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) return;
    const r = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const hit = ray.intersectObjects(drawn, false)[0];
    const lines = hit ? describe(extrasAt(hit)) : [];
    pick.replaceChildren(...lines.map((t) => Object.assign(document.createElement('p'), { textContent: t })));
    pick.hidden = !lines.length;
    render();
  });

  let framed = false;
  function resize() {
    const w = stage.clientWidth || 1;
    const h = stage.clientHeight || 1;
    renderer.setSize(w, h, false);
    labels.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (!framed) {
      framed = true;
      frame();
    }
    resolution.set(w, h); // CSS pixels: outline widths are CSS pixels on any screen
    lineMat.resolution.copy(resolution);
    dashMat?.resolution.copy(resolution);
    render();
  }
  function render() {
    const t0 = performance.now();
    updateTags();
    renderer.render(scene, camera);
    labels.render(scene, camera);
    state.renderMs = Math.round(performance.now() - t0);
  }
  state.timeRender = () => {
    // one synchronous frame (readPixels waits for the GPU): for performance checks
    const t0 = performance.now();
    render();
    const gl = renderer.getContext();
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    return Math.round(performance.now() - t0);
  };
  controls.addEventListener('change', render);
  new ResizeObserver(resize).observe(stage);
  resize();
  requestAnimationFrame(() => {
    render();
    state.ready = true;
  });
}

if (typeof window !== 'undefined' && typeof document !== 'undefined' && !globalThis.__STYLIZED_NO_AUTOSTART) {
  main().catch((e) => {
    const status = document.getElementById('status');
    if (status) {
      status.textContent = `出错了：${e.message ?? e}`;
      status.hidden = false;
    }
    window.__stylized = { ready: true, error: String(e) };
  });
}
