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
// The look is a theme (web/themes/<id>.json, see themes.js and docs/ART_DIRECTION.md): palette mapping,
// clay, ink, outlines, toon steps, lights, background. A theme only restyles: the GLB names every colour
// it uses (scene extras style.colors) and its atlases are lossless, so the facade textures are remapped
// texel by texel, and a theme under which clay could pass for a photographed surface is refused.
//
// Query parameters: model=<GLB url> (required), bearing=<deg, compass bearing from the building to
// the camera; default: the side with the most photographed facades>, elev=<deg>, zoom=<factor>,
// ui=0 (hide the panel, for screenshots), outline=<px> (overrides the theme's outline width),
// theme=<id> (default: the original look).
//
// The pure helpers at the top are exported for node tests; three.js is imported lazily so that
// importing this module outside a browser does not need it.

import {
  DEFAULT_THEME,
  THEME_ID_RE,
  buildColorMap,
  loadTheme,
  remapPixels,
  resolveTheme,
  themeCss,
  validateTheme,
} from './themes.js';

export const TOON_STEPS = DEFAULT_THEME.toon.steps;
export const INK = 0x2d2a32; // = crowd3d.stylize.INK_RGB (the default theme's ink)
export const CLAY = 0xe4e8ee; // = crowd3d.stylize.CLAY_RGB: the clay texels of the atlases (each theme remaps them to its clay.wall)
export const LOCKED_LABEL = '待解锁';
export const PROVENANCE_TEXT = {
  'photo-observed': '照片实拍',
  'observed-fill': '实拍墙面的底色补洞',
  'osm-footprint': 'OSM 轮廓',
  'sfm-points': '重建点云实测',
  'osm-height': 'OSM 高度标签',
  'wikidata-height': 'Wikidata 高度',
  inferred: '推断（同一体块别处实测，这里没测到）',
  'default-style': '默认样式（未拍到）',
  // structure-first parts (crowd3d/structure.py, docs/STRUCTURE_FIRST.md §2.7)
  'osm-s3db': 'OpenStreetMap 志愿者测绘的分件',
  'osm-levels': 'OSM 层数换算（每层 3 米）',
  'osm-roof-angle': 'OSM 屋顶坡度换算',
  's3db-shape-default': '屋顶坡度按惯例（地图只标了屋顶形状）',
  'osm-s3db-support': '由压在上面的分件推出',
  'osm-default-ground': '地图默认（从地面起）',
  curator: '编辑核对',
  'photo-fitted': '照片实测（改了地图值）',
  // D15: at least two photo channels agreed; the map's value is kept beside it and the change can be undone
  'photo-fitted (auto)': '照片量出（自动采用）',
  // D25: strong photo evidence corrected the map (the full photo value, or only its guaranteed bound); the map's value
  // is kept beside it and the change can be undone; OpenStreetMap itself is never edited
  'photo-corrected (auto)': '照片纠正地图（自动）',
  'photo-corrected (auto, lower bound)': '照片纠正地图（自动，下界）',
};
export const CORRECTED_LABEL = '照片纠正了地图';
export const AUTO_SOURCE = 'photo-fitted (auto)';
export const AUTO_ADOPTED_LABEL = '照片量出（自动采用）';
export const GEOMETRY_TEXT = {
  map: '形状：OpenStreetMap 志愿者测绘',
  'map-convention': '形状：地图事实加惯例（屋顶坡度按惯例）',
  'photo-verified': '形状：已由照片核验',
  'photo-fitted': '形状：照片实测',
  'photo-corrected': `形状：照片纠正了地图`,
  disputed: '形状：OpenStreetMap 志愿者测绘',
  curator: '形状：编辑核对',
  carved: '高度由照片测得，屋顶形状未知',
  stub: '高度待定（地图没有高度）',
};
// dense parts models have hundreds of wall faces: only the largest locked ones get a dashed frame
// facade relief the photos measured (crowd3d.relief element kinds)
export const RELIEF_KIND_TEXT = {
  arch: '拱', 'arched-bay': '拱券开间', bay: '凹开间', panel: '凹板', window: '窗洞', 'arched-window': '拱窗',
  door: '门洞', 'arched-door': '拱门', setback: '退台', groove: '凹槽', band: '腰线', plinth: '勒脚',
  'projecting-zone': '凸出带', pilaster: '壁柱', buttress: '扶壁', 'corner-pier': '角柱',
};
// facade relief provenance (crowd3d.relief): measured by both photo splits with a 3-D line at an edge, or (D24, the
// product owner's decision of 2026-10-24) built on the photos' point cloud alone where the wall's 3-D lines are at chance
export const RELIEF_MEASURED = 'photo-measured';
export const RELIEF_D24 = 'photo-measured (points, D24)';
export const RELIEF_D24_LABEL = '照片点云量出（D24）';
export const MAX_FRAMES = 48;
export const MIN_FRAME_AREA = 12; // m2 (three.js units are metres)

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
    outlinePx: num(params, 'outline', null, 0, 8), // null: the theme's
    ui: params.get('ui') !== '0',
    theme: THEME_ID_RE.test(params.get('theme') ?? '') ? params.get('theme') : 'default',
  };
}

/** The ids the theme picker offers: those of themes/index.json, plus the current one if missing. */
export function pickerChoices(index, current) {
  const ids = Array.isArray(index?.themes) ? index.themes.filter((id) => THEME_ID_RE.test(id)) : [];
  if (!ids.includes('default')) ids.unshift('default');
  if (THEME_ID_RE.test(current) && !ids.includes(current)) ids.push(current);
  return ids;
}

/** location.search for the same view in another theme (the default theme drops the parameter). */
export function themeSearch(search, id) {
  const params = new URLSearchParams(search);
  if (id === 'default') params.delete('theme');
  else params.set('theme', id);
  const s = params.toString();
  return s ? `?${s}` : '';
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
    reliefElements: wallParts.reduce(
      (s, e) => s + (Array.isArray(e.relief) ? e.relief.filter((r) => r && r.built && isReliefMeasured(r)).length : 0), 0),
  };
}

/** A relief record the photos measured: with a 3-D line at an edge, or on the point cloud alone (D24, with its evidence). */
export function isReliefMeasured(r) {
  return Boolean(r) && (r.provenance === RELIEF_MEASURED || (r.provenance === RELIEF_D24 && r.d24 && typeof r.d24 === 'object'));
}

/** Lines describing a picked face. */
export function describe(extras) {
  if (!extras || !extras.kind) return [];
  const kind = { wall: '立面', roof: '屋顶', 'roof-patch': '屋顶实拍区', 'roof-cut': '屋顶切面', corners: '转角', base: '底面' }[extras.kind] ?? extras.kind;
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
  if (extras.fill?.holes > 0) {
    const m2 = Math.round(extras.fill.area_m2 * 100) / 100;
    lines.push(`底色补洞：${extras.fill.holes} 处，共 ${m2} m²（墙面中间被挡住的小块，用这面墙的实测底色补齐，不算实拍覆盖）`);
  }
  if (Array.isArray(extras.coverage) && extras.coverage.length && extras.kind !== 'wall') {
    lines.push(`点云覆盖：${extras.coverage.map(pct).join(' / ')}`);
  }
  if (Array.isArray(extras.photos) && extras.photos.length) lines.push(`来自 ${extras.photos.length} 张照片`);
  const relief = Array.isArray(extras.relief) ? extras.relief.filter(isReliefMeasured) : [];
  if (relief.length) {
    // structure the photos measured on this very wall; nothing copied. Two photo halves each and a 3-D line at an
    // edge; or (D24) both photo splits and the raw points alone, where this wall's 3-D lines are no better than chance
    const kinds = (rs) => {
      const counts = new Map();
      for (const r of rs) counts.set(r.kind, (counts.get(r.kind) ?? 0) + 1);
      return [...counts].map(([k, n]) => `${RELIEF_KIND_TEXT[k] ?? k} ${n}`).join('、');
    };
    const built = relief.filter((r) => r.built);
    const lined = built.filter((r) => r.provenance === RELIEF_MEASURED);
    const d24 = built.filter((r) => r.provenance === RELIEF_D24);
    if (lined.length) lines.push(`立面起伏：照片实测 ${lined.length} 处（${kinds(lined)}），每处两半照片各自量到，并有三维直线核对`);
    if (d24.length) {
      lines.push(`立面起伏：${RELIEF_D24_LABEL} ${d24.length} 处（${kinds(d24)}），两种照片分法都量到，原始点重量对得上；`
        + '这面墙的三维直线到处都是（处于偶然水平），没有直线核对');
    }
    if (relief.length > built.length) lines.push(`另实测 ${relief.length - built.length} 处不单独建模（如两个凹开间之间的壁柱就是墙面本身）`);
  }
  if (extras.part) {
    // a structure part: where its shape comes from (geometry and appearance are separate axes); a 'height pending'
    // stub (D13: height default-style) always reads as one, whatever state an older or edited model gives it
    const geometry = p.height === 'default-style' ? 'stub' : extras.geometry;
    // D15: a value two photo channels agreed on reads 照片量出（自动采用）, never as a mapped or a curator's shape;
    // an outline the frame adoption scaled and shifted (D15 ⑥) is not OpenStreetMap's as drawn either
    const outlineAdopted = Boolean(extras.outline_adopted) || p.outline === AUTO_SOURCE;
    const framed = outlineAdopted ? `（轮廓按照片整体缩放、平移：${AUTO_ADOPTED_LABEL}）` : '';
    if (geometry === 'photo-fitted' && p.height === AUTO_SOURCE) lines.push(`形状：${AUTO_ADOPTED_LABEL}${framed}`);
    else if (GEOMETRY_TEXT[geometry]) lines.push(`${GEOMETRY_TEXT[geometry]}${framed}`);
    const metres = (v) => (Number.isFinite(v) ? `${Math.round(v * 100) / 100} m` : '?');
    const oa = extras.outline_adopted;
    if (oa && typeof oa === 'object') {
      const k = Number.isFinite(oa.k) ? `×${Math.round(oa.k * 1000) / 1000}` : '';
      const shift = Array.isArray(oa.t) && oa.t.every(Number.isFinite) ? `、平移 ${metres(Math.hypot(oa.t[0], oa.t[1]))}` : '';
      lines.push(`轮廓：${AUTO_ADOPTED_LABEL}：整栋缩放 ${k}${shift}（地图原轮廓宽 ${metres(oa.width_map_m)}，现 ${metres(oa.width_m)}，可撤销）`);
    }
    const what = { height: '顶高', min_height: '底高', roof_height: '屋顶高' };
    for (const a of Array.isArray(extras.adopted) ? extras.adopted : []) {
      const how = a.carried ? '随下面的分件一起移动' : AUTO_ADOPTED_LABEL;
      lines.push(`${what[a.param] ?? a.param}：${how} ${metres(a.value)}（地图原值 ${metres(a.map)}，可撤销）`);
    }
    // D25 (照片纠正了地图): old -> new for every corrected value; a bound reads 'at least'; a block's roof shape is unknown
    const PFULL = PROVENANCE_TEXT['photo-corrected (auto)'];
    const PLB = PROVENANCE_TEXT['photo-corrected (auto, lower bound)'];
    for (const c of Array.isArray(extras.corrected) ? extras.corrected : []) {
      if (!c || !Number.isFinite(c.value)) continue;
      const name = what[c.param] ?? c.param;
      const range = Array.isArray(c.range) && c.range.every(Number.isFinite)
        ? `（范围 ${metres(c.range[0])}–${metres(c.range[1])}）` : '';
      if (c.carried) lines.push(`${name}：随下面被纠正的分件一起移动 ${metres(c.value)}（地图原值 ${metres(c.map)}，可撤销）`);
      else if (c.kind === 'd25-block') lines.push(`照片量出：至少 ${metres(c.value)}（屋顶形状未知；地图上高度待定 ${metres(c.map)}，可撤销）`);
      else if (c.adopted === 'lower bound') {
        const pv = Number.isFinite(c.photo_value) ? `，照片值 ${metres(c.photo_value)}` : '';
        lines.push(`${PLB}：${name}至少 ${metres(c.value)}（地图 ${metres(c.map)}${pv}，可撤销）`);
      } else if (c.adopted === 'upper bound') {
        lines.push(`${PLB}：${name}至多 ${metres(c.value)}（地图 ${metres(c.map)}，可撤销）`);
      } else lines.push(`${PFULL}：${name} 地图 ${metres(c.map)} → 照片 ${metres(c.value)}${range}（可撤销）`);
    }
    const tilt = extras.tilt;
    if (tilt && Number.isFinite(tilt.tilt_deg)) {
      const az = Number.isFinite(tilt.azimuth_deg) ? `，朝 ${Math.round(tilt.azimuth_deg)}°` : '';
      lines.push(`倾斜：照片量出 ${Math.round(tilt.tilt_deg * 10) / 10}°${az}（地图上是直的，可撤销）`);
    }
    if (extras.roof?.fallback_from) lines.push(`屋顶：地图标的是 ${extras.roof.fallback_from}，搭不出来，画成 ${extras.roof.shape}`);
  }
  return lines;
}

/**
 * The roof's structural lines a model carries (scene extras structure_lines: ridges, hips, valleys and rakes of the
 * drawn parts, in glTF axes; eaves are left to the crease lines) as flat xyz arrays for line segments: solid, and
 * dashed for a conventional roof shape (D17: roof:shape tagged without a roof height). Drawn whatever the angle
 * between their faces (a low hip stays under the crease threshold). Malformed rows are skipped.
 */
export function structureLineSegments(sl) {
  const solid = [];
  const dashed = [];
  const types = Array.isArray(sl?.types) ? sl.types : [];
  for (const row of Array.isArray(sl?.lines) ? sl.lines : []) {
    if (!Array.isArray(row) || row.length < 8 || !row.slice(0, 6).every(Number.isFinite)) continue;
    if (types[row[6]] === 'eave') continue; // where roof meets wall: a crease line already
    (row[7] ? dashed : solid).push(...row.slice(0, 6));
  }
  return { solid, dashed };
}

/** The locked walls that get a dashed frame: at least minArea, the largest first, at most max. */
export function pickFrames(walls, max = MAX_FRAMES, minArea = MIN_FRAME_AREA) {
  return walls.filter((w) => w.locked && w.area >= minArea).sort((a, b) => b.area - a.area).slice(0, max);
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

/**
 * The look a model is shown in: the requested theme if it passes the validator against the model's colour
 * table, else the default (with a note pushed to `notes`). Every theme, the default included, gets a colour
 * map over that table, so the atlases' clay is always the theme's clay. table: style.colors of the GLB (or
 * the viewer's copy), null when neither loaded; requestedId: the ?theme= value, for the note.
 */
export function pickLook(theme, table, requestedId, shadeFactor, notes) {
  let check = table ? validateTheme(theme, table) : { ok: false, errors: ['没有颜色表，无法换画风'] };
  if (!check.ok && theme !== DEFAULT_THEME) {
    notes.push(`画风“${theme.name ?? requestedId}”没有通过校验，已改用默认画风：${check.errors[0]}`);
    theme = DEFAULT_THEME;
    check = table ? validateTheme(theme, table) : check;
  }
  if (table && !check.ok) notes.push(`默认画风没有通过这个模型颜色表的校验：${check.errors[0]}`);
  const look = resolveTheme(theme);
  const colors = table ? buildColorMap(look, table, shadeFactor) : null;
  return { look, colors, restyle: !!colors && !colors.identity, check };
}

function decorate(material, { hatchShade = null, gradient = null }) {
  // faint diagonal stripes in world space on locked faces ("not unlocked yet"); a theme's vertical
  // gradient (a multiplicative tint by height) on every face of the building
  if (hatchShade === null && !gradient) return;
  material.onBeforeCompile = (shader) => {
    if (gradient) Object.assign(shader.uniforms, gradient);
    let colour = '#include <color_fragment>';
    if (hatchShade !== null) {
      colour += '\nfloat hatch = fract((vHatchPos.x + vHatchPos.y - vHatchPos.z) / 1.8);\n'
        + `diffuseColor.rgb *= mix(1.0, ${hatchShade.toFixed(3)}, step(0.72, hatch));`;
    }
    if (gradient) {
      colour += '\nfloat gradT = clamp((vHatchPos.y - uGradY0) / max(uGradY1 - uGradY0, 1e-3), 0.0, 1.0);\n'
        + 'diffuseColor.rgb *= mix(vec3(1.0), mix(uGradBottom, uGradTop, gradT), uGradStrength);';
    }
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vHatchPos;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvHatchPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vHatchPos;'
        + (gradient ? '\nuniform vec3 uGradTop;\nuniform vec3 uGradBottom;\nuniform float uGradStrength;\nuniform float uGradY0;\nuniform float uGradY1;' : ''))
      .replace('#include <color_fragment>', colour);
  };
  material.customProgramCacheKey = () => `crowd3d-${hatchShade === null ? 'n' : hatchShade.toFixed(3)}-${gradient ? 'g' : 'n'}`;
}

function outlineMaterial(THREE, px, resolution, color) {
  // inverted hull, offset in clip space so the silhouette is `px` pixels wide at any distance
  return new THREE.ShaderMaterial({
    uniforms: { thickness: { value: px }, resolution: { value: resolution }, ink: { value: new THREE.Color(color) } },
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
      void main() {
        gl_FragColor = vec4(ink, 1.0);
        #include <colorspace_fragment>
      }`,
    side: THREE.BackSide,
  });
}

function remapTexture(THREE, tex, colors) {
  // the atlas through the theme's colour table, texel by texel (atlases are lossless: exact matches)
  const img = tex.image;
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const stats = remapPixels(data.data, colors.map, colors.keys);
  ctx.putImageData(data, 0, 0);
  const out = new THREE.Texture(canvas);
  for (const k of ['flipY', 'colorSpace', 'magFilter', 'minFilter', 'wrapS', 'wrapT', 'generateMipmaps']) out[k] = tex[k];
  out.needsUpdate = true;
  return { texture: out, stats };
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function fillThemePicker(select, current) {
  let index = null;
  try {
    index = await fetchJson('themes/index.json');
  } catch {
    // the picker then offers the default and the current theme only
  }
  const ids = pickerChoices(index, current);
  const names = await Promise.all(ids.map(async (id) => {
    if (id === 'default') return DEFAULT_THEME.name;
    try {
      return (await fetchJson(`themes/${id}.json`)).name ?? id;
    } catch {
      return id;
    }
  }));
  select.replaceChildren(...ids.map((id, i) => Object.assign(document.createElement('option'), {
    value: id, textContent: names[i], selected: id === current,
  })));
  select.disabled = false;
  select.addEventListener('change', () => {
    location.search = themeSearch(location.search, select.value);
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
  const state = (window.__stylized = { ready: false, error: null, stats: null, theme: null });

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

  const requested = await loadTheme(opts.theme, fetchJson);
  let gltf;
  try {
    gltf = await new GLTFLoader().loadAsync(opts.model);
  } catch (e) {
    return fail(`模型加载失败：${e.message ?? e}`);
  }
  const model = gltf.scene;
  const info = model.userData ?? {};
  model.updateMatrixWorld(true);

  // the theme: checked against the colour table the model carries (older models: the viewer's copy);
  // one that fails (or does not load) gives way to the default look, with a note in the panel. Every theme,
  // the default included, restyles the atlases through that table (the default only changes their clay)
  const notes = requested.error ? [requested.error] : [];
  const theme = requested.theme;
  let table = info.style?.colors ?? null;
  if (!table) {
    try {
      table = (await fetchJson('themes/palette.json')).colors;
    } catch (e) {
      notes.push(`颜色表加载失败：${e.message ?? e}`);
    }
  }
  const { look, colors, restyle, check } = pickLook(theme, table, opts.theme, info.style?.shade_factor, notes);
  state.theme = { id: look.id, requested: opts.theme, notes, minClayDeltaE: check?.minClayDeltaE ?? null,
    texels: { exact: 0, near: 0, unknown: 0 } };
  const css = themeCss(look);
  document.body.style.background = css.background;
  document.documentElement.style.setProperty('--clay', css.clay);

  // intensities in three.js physical units: albedo x (hemisphere ~0.4 + sun ~0.65 x toon step)
  const L = look.lights;
  scene.add(new THREE.HemisphereLight(L.hemisphere.sky, L.hemisphere.ground, L.hemisphere.intensity));
  const sun = new THREE.DirectionalLight(L.sun.color, L.sun.intensity);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.bias = -0.0015;
  sun.shadow.normalBias = 0.08; // no shadow acne stripes on large flat walls
  scene.add(sun, sun.target);

  const steps = look.toon.steps;
  const gradient = new THREE.DataTexture(toonGradientData(steps), steps.length, 1, THREE.RGBAFormat);
  gradient.minFilter = gradient.magFilter = THREE.NearestFilter;
  gradient.needsUpdate = true;

  const box = new THREE.Box3().setFromObject(model); // (not added to the scene: `drawn` is)
  const tint = look.gradient && {
    uGradTop: { value: new THREE.Color(look.gradient.top) },
    uGradBottom: { value: new THREE.Color(look.gradient.bottom) },
    uGradStrength: { value: look.gradient.strength },
    uGradY0: { value: box.min.y },
    uGradY1: { value: box.max.y },
  };
  const themed = new Map(); // source atlas -> its remapped copy
  const themedMap = (tex) => {
    if (!restyle || !tex) return tex;
    if (!themed.has(tex)) {
      const { texture, stats } = remapTexture(THREE, tex, colors);
      for (const k of Object.keys(stats)) state.theme.texels[k] += stats[k];
      themed.set(tex, texture);
    }
    return themed.get(tex);
  };
  const flatColour = (old) => {
    // a flat palette colour: by the name the material carries, else by its own value
    if (!restyle) return old.color;
    let rgb = old.userData?.color ? colors.byName.get(old.userData.color) : null;
    const key = old.color.getHex(THREE.SRGBColorSpace);
    if (!rgb && colors.map.has(key)) rgb = [(colors.map.get(key) >> 16) & 255, (colors.map.get(key) >> 8) & 255, colors.map.get(key) & 255];
    return rgb ? new THREE.Color().setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace) : old.color;
  };

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
        color: old.map ? 0xffffff : flatColour(old),
        map: themedMap(old.map ?? null),
        gradientMap: gradient,
        // shadows from the faces turned to the sun (three.js casts single-sided meshes from their back faces,
        // which dropped dark blobs onto walls in the light behind thin parts: lanterns, spires)
        shadowSide: THREE.FrontSide,
      });
      const locked = ex.locked && !old.map;
      if (locked) mat.color = new THREE.Color(ex.kind === 'roof' ? look.clay.roof : look.clay.face);
      decorate(mat, { hatchShade: locked ? look.clay.hatch_shade : null, gradient: tint });
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

  // silhouette + crease lines from one welded copy of the whole model (a theme may have none)
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
  const outline = look.outline;
  const silhouettePx = outline ? (opts.outlinePx ?? outline.width_px) : 0;
  const creasePx = opts.outlinePx !== null ? Math.max(1, opts.outlinePx * 0.7) : outline?.crease_width_px;
  if (silhouettePx > 0) {
    scene.add(new THREE.Mesh(welded, outlineMaterial(THREE, silhouettePx, resolution, outline.silhouette_color)));
  }
  let lineMat = null;
  if (outline) {
    const creases = new LineSegmentsGeometry().fromEdgesGeometry(new THREE.EdgesGeometry(welded, 35));
    lineMat = new LineMaterial({ color: outline.crease_color, linewidth: creasePx, resolution });
    scene.add(new LineSegments2(creases, lineMat));
  }
  // the roof's ridges, hips and valleys as the model lists them (a conventional roof shape dashed, D17)
  const roofLines = structureLineSegments(info.structure_lines);
  let roofDashMat = null;
  if (outline && roofLines.solid.length) scene.add(new LineSegments2(new LineSegmentsGeometry().setPositions(roofLines.solid), lineMat));
  if (outline && roofLines.dashed.length) {
    roofDashMat = new LineMaterial({ color: outline.crease_color, linewidth: creasePx, dashed: true, dashSize: 0.6, gapSize: 0.45, resolution });
    const dl = new LineSegments2(new LineSegmentsGeometry().setPositions(roofLines.dashed), roofDashMat);
    dl.computeLineDistances();
    scene.add(dl);
  }
  // locked walls: a dashed placeholder frame ("to be unlocked")
  const frames = pickFrames(walls).flatMap((w) => insetFrame(w.corners, w.normal));
  let dashMat = null;
  if (frames.length) {
    const g = new LineSegmentsGeometry().setPositions(frames);
    const framePx = Math.max(1, (opts.outlinePx ?? outline?.width_px ?? DEFAULT_THEME.outline.width_px) * 0.8);
    dashMat = new LineMaterial({ color: look.locked_frame, linewidth: framePx, dashed: true,
      dashSize: 0.9, gapSize: 0.6, resolution });
    const dashes = new LineSegments2(g, dashMat);
    dashes.computeLineDistances();
    scene.add(dashes);
  }

  // ground disc and framing
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(4, 0.5 * Math.hypot(size.x, size.y, size.z));
  const footR = 0.5 * Math.hypot(size.x, size.z);
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(footR * 1.3, 96),
    new THREE.MeshToonMaterial({ color: look.ground.color, gradientMap: gradient }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(center.x, -0.02, center.z);
  ground.receiveShadow = true;
  const rim = new THREE.Mesh(
    new THREE.RingGeometry(footR * 1.3, footR * 1.37, 96),
    new THREE.MeshBasicMaterial({ color: look.ground.rim }),
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
  const themeNote = document.getElementById('theme-note');
  themeNote.textContent = notes.join('；');
  themeNote.hidden = !notes.length;
  for (const n of notes) console.warn(n);
  panel.hidden = !opts.ui;
  if (opts.ui) fillThemePicker(document.getElementById('theme-pick'), look.id).catch(() => {});

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
    lineMat?.resolution.copy(resolution);
    dashMat?.resolution.copy(resolution);
    roofDashMat?.resolution.copy(resolution);
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
