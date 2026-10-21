// three.js point-cloud viewer for the PLY models the API serves.
//
// Models are in ENU metres at the landmark centre (x = east, y = north, z = up).
// three.js is y-up, so ENU (x, y, z) maps to (x, z, -y): up stays up and north is -Z.

import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { fitView, framingPoints, viewDirection } from './framing.js';

const ENU_TO_THREE = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
const MAX_SAMPLES = 20000;
const DEFAULT_VIEW_BEARING_DEG = 140; // south-east, the usual "map" viewpoint
const VIEW_ELEVATION_DEG = 30;

export function enuToThree([x, y, z]) {
  return new THREE.Vector3(x, z, -y);
}

/** 1-2-5 rounding up: 13 -> 20, 0.3 -> 0.5. */
function niceCeil(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 5, 10]) if (v <= m * p) return m * p;
  return 10 * p;
}

function percentile(sorted, q) {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))))];
}

// SfM clouds carry stray points far from the building; framing on the 2-98 %
// range keeps the camera on the model instead of on the outliers.
function robustBox(position) {
  const step = Math.max(1, Math.floor(position.count / MAX_SAMPLES));
  const axes = [[], [], []];
  for (let i = 0; i < position.count; i += step) {
    axes[0].push(position.getX(i));
    axes[1].push(position.getY(i));
    axes[2].push(position.getZ(i));
  }
  const lo = new THREE.Vector3();
  const hi = new THREE.Vector3();
  axes.forEach((values, k) => {
    values.sort((a, b) => a - b);
    lo.setComponent(k, percentile(values, 0.02));
    hi.setComponent(k, percentile(values, 0.98));
  });
  return new THREE.Box3(lo, hi);
}

function textSprite(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = color;
  ctx.font = 'bold 44px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 34);
  const map = new THREE.CanvasTexture(canvas);
  map.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Sprite(new THREE.SpriteMaterial({ map, depthTest: false }));
}

function boxCorners(box) {
  const corners = [];
  for (const x of [box.min.x, box.max.x])
    for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
  return corners;
}

function disposeTree(object) {
  object.traverse((o) => {
    o.geometry?.dispose();
    const materials = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
    for (const m of materials) {
      m.map?.dispose();
      m.dispose();
    }
  });
}

export class ModelViewer {
  /** Throws if WebGL is unavailable. */
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x000000, 0); // the container's CSS background shows through, so themes just work
    const canvas = this.renderer.domElement;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', '三维点云模型，可拖动旋转、缩放');
    container.append(canvas);

    // Screen-space compass that turns with the camera, so north stays readable from any angle.
    this.northBadge = document.createElement('div');
    this.northBadge.className = 'viewer-north';
    this.northBadge.setAttribute('aria-hidden', 'true');
    this.northBadge.innerHTML =
      '<svg viewBox="0 0 44 44"><text x="22" y="8" text-anchor="middle" dominant-baseline="central">北</text>' +
      '<path class="needle-n" d="M22 12 27 22 17 22Z"/><path class="needle-s" d="M17 22 27 22 22 32Z"/></svg>';
    container.append(this.northBadge);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 5000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.maxPolarAngle = Math.PI * 0.49; // stay above the ground plane
    this.controls.addEventListener('change', () => this.render());

    this.cloud = null;
    this.helpers = null;
    this.cameraMarks = null;
    this.cameraPositions = [];
    this.camerasVisible = true;
    this.radius = 10;
    this.loadToken = 0;
    this.loader = new PLYLoader();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  /**
   * Load a PLY url; resolves to {points} or null when a newer load superseded it.
   * viewFromDeg: compass bearing (from the model towards the camera) to look from, e.g. a facade's sector.
   * cameras: the model's estimated photo poses (see setCameras); while they are shown, the first view
   * also takes them in, since photographers stand well away from the facade.
   */
  async load(url, { viewFromDeg = DEFAULT_VIEW_BEARING_DEG, cameras = [] } = {}) {
    const token = ++this.loadToken;
    const geometry = await this.loader.loadAsync(url);
    if (token !== this.loadToken) {
      geometry.dispose();
      return null;
    }
    geometry.applyMatrix4(ENU_TO_THREE);
    const position = geometry.getAttribute('position');
    if (!position || position.count === 0) {
      geometry.dispose();
      throw new Error('模型中没有点');
    }

    const box = robustBox(position);
    const size = box.getSize(new THREE.Vector3());
    this.radius = Math.max(1, size.length() / 2);
    const material = new THREE.PointsMaterial({
      // Sparse SfM clouds read as faint dust with hairline points; ~radius/50 makes surfaces legible.
      size: Math.max(0.03, this.radius / 50),
      sizeAttenuation: true,
      vertexColors: geometry.hasAttribute('color'),
      color: geometry.hasAttribute('color') ? 0xffffff : 0x7a8da8,
    });

    if (this.cloud) {
      this.scene.remove(this.cloud);
      disposeTree(this.cloud);
    }
    this.cloud = new THREE.Points(geometry, material);
    this.scene.add(this.cloud);

    this.setCameras(cameras);
    const positions = this.camerasVisible ? this.cameraPositions.map((p) => p.toArray()) : [];
    const extra = framingPoints(positions, box.getCenter(new THREE.Vector3()).toArray(), this.radius).map(
      (p) => new THREE.Vector3(...p),
    );
    this.buildHelpers(box, viewFromDeg, box.clone().setFromPoints([box.min, box.max, ...extra]));
    this.frame(box, viewFromDeg, extra);
    this.container.dataset.points = String(position.count);
    this.container.dataset.framedCameras = String(extra.length);
    this.render();
    return { points: position.count };
  }

  /** Estimated camera poses: [{enu: [x, y, z], heading_deg}] -> dots plus short heading ticks. */
  setCameras(cameras) {
    if (this.cameraMarks) {
      this.scene.remove(this.cameraMarks);
      disposeTree(this.cameraMarks);
      this.cameraMarks = null;
    }
    const valid = cameras.filter((c) => Array.isArray(c.enu) && c.enu.length === 3);
    this.cameraPositions = valid.map((c) => enuToThree(c.enu));
    if (valid.length) {
      const dots = [];
      const ticks = [];
      const len = Math.max(1, this.radius * 0.12);
      for (const c of valid) {
        const p = enuToThree(c.enu);
        dots.push(p.x, p.y, p.z);
        if (Number.isFinite(c.heading_deg)) {
          const a = (c.heading_deg * Math.PI) / 180;
          const q = p.clone().add(enuToThree([Math.sin(a) * len, Math.cos(a) * len, 0]));
          ticks.push(p.x, p.y, p.z, q.x, q.y, q.z);
        }
      }
      const color = 0xf06a2b;
      const group = new THREE.Group();
      const dotGeo = new THREE.BufferGeometry();
      dotGeo.setAttribute('position', new THREE.Float32BufferAttribute(dots, 3));
      group.add(new THREE.Points(dotGeo, new THREE.PointsMaterial({ color, size: 6, sizeAttenuation: false })));
      if (ticks.length) {
        const tickGeo = new THREE.BufferGeometry();
        tickGeo.setAttribute('position', new THREE.Float32BufferAttribute(ticks, 3));
        group.add(new THREE.LineSegments(tickGeo, new THREE.LineBasicMaterial({ color })));
      }
      group.visible = this.camerasVisible;
      this.cameraMarks = group;
      this.scene.add(group);
    }
    this.container.dataset.cameras = String(valid.length);
    this.render();
  }

  setCamerasVisible(visible) {
    this.camerasVisible = visible;
    if (this.cameraMarks) this.cameraMarks.visible = visible;
    this.render();
  }

  // Ground grid centred on the landmark origin and reaching under everything in view (gridBox), plus a north
  // arrow lying on it. The arrow sits in front of the model and to the right in the initial view, so the
  // model does not hide it and it stays in frame.
  buildHelpers(box, fromBearingDeg, gridBox = box) {
    if (this.helpers) {
      this.scene.remove(this.helpers);
      disposeTree(this.helpers);
    }
    const g = gridBox;
    const reach = Math.max(Math.abs(g.min.x), Math.abs(g.max.x), Math.abs(g.min.z), Math.abs(g.max.z), 5);
    const half = niceCeil(reach * 1.15);
    const cell = niceCeil(half / 8);
    const grid = new THREE.GridHelper(half * 2, Math.round((half * 2) / cell), 0x8a94a6, 0x8a94a6);
    grid.material.transparent = true;
    grid.material.opacity = 0.35;

    const group = new THREE.Group();
    group.add(grid);
    const arrowLen = this.radius * 0.3;
    // The camera is at fromBearingDeg looking back at the model: bearing - 90 is to its right, - 60 is front-right.
    const side = THREE.MathUtils.degToRad(fromBearingDeg - 60);
    const center = box.getCenter(new THREE.Vector3());
    const north = new THREE.Vector3(0, 0, -1);
    // Centred on its anchor, so it stays in frame also when north points at the camera.
    const base = new THREE.Vector3(center.x, 0.05, center.z)
      .add(enuToThree([Math.sin(side), Math.cos(side), 0]).multiplyScalar(this.radius * 0.95))
      .addScaledVector(north, -arrowLen / 2);
    group.add(new THREE.ArrowHelper(north, base, arrowLen, 0xd9473f, arrowLen * 0.3, arrowLen * 0.2));
    const label = textSprite('N', '#d9473f');
    label.position.copy(base).addScaledVector(north, arrowLen * 1.25);
    label.scale.setScalar(this.radius * 0.1);
    group.add(label);
    this.helpers = group;
    this.scene.add(group);
  }

  // The cloud alone: a sphere around its box fits from every orbit angle. With extra points (the cameras),
  // fit the cloud box corners and the points tightly into this view instead, so the model stays large;
  // the orbit then turns around the middle of model and cameras.
  frame(box, fromBearingDeg = DEFAULT_VIEW_BEARING_DEG, extra = []) {
    const dir = new THREE.Vector3(...viewDirection(fromBearingDeg, VIEW_ELEVATION_DEG));
    let center = box.getCenter(new THREE.Vector3());
    let distance = (this.radius / Math.sin((this.camera.fov * Math.PI) / 360)) * 1.05;
    if (extra.length) {
      const points = [...boxCorners(box), ...extra].map((p) => p.toArray());
      const fit = fitView(points, dir.toArray(), this.camera.fov, this.camera.aspect);
      center = new THREE.Vector3(...fit.target);
      distance = fit.distance;
    }
    this.camera.position.copy(center).addScaledVector(dir, distance);
    this.camera.near = distance / 200;
    this.camera.far = distance * 20;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(center);
    this.controls.update();
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
    // Azimuth 0 looks from the south towards north (-Z), i.e. north is screen-up.
    this.northBadge.firstChild.style.transform = `rotate(${this.controls.getAzimuthalAngle()}rad)`;
  }

  dispose() {
    this.loadToken++;
    this.resizeObserver.disconnect();
    this.controls.dispose();
    disposeTree(this.scene);
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.northBadge.remove();
  }
}
