// Fetch helpers for the Crowd3D HTTP API (shapes in the prototype spec, "HTTP API"),
// plus Chinese texts for the codes the API returns.

export const REASON_TEXT = {
  unreadable_image: '无法读取图片',
  no_exif: '没有 EXIF 信息，请上传相机原图（截图、编辑过或经聊天软件发送的图片会丢失）',
  no_gps: '缺少GPS定位',
  no_landmark: '附近没有地标',
  too_far: '距离地标太远',
  facing_away: '镜头没有朝向建筑',
  too_small: '分辨率太低',
  too_blurry: '照片太模糊',
  duplicate: '重复照片',
  too_large: '文件过大',
  // D31; "50 米" is Settings.subject_max_distance_m (tests/test_validation.py checks that they agree)
  not_subject: '离建筑超过 50 米，建筑又不是画面主体：请走近拍，或让建筑占满画面',
};

export const WARNING_TEXT = {
  no_heading: '缺少指南针朝向',
  magnetic_heading: '使用磁北朝向',
  subject_unchecked: '没有检查建筑是不是照片主体',   // the landmark has no bound OSM building outline
};

export const JOB_STATUS_TEXT = { queued: '排队中', running: '生成中', done: '已完成', failed: '失败' };

// Job kinds besides the models' own (MODEL_KIND_TEXT): a stylize job makes a published model's stylized GLB.
export const JOB_KIND_TEXT = { stylize: '风格化' };

// Job errors are English free text from crowd3d/reconstruct.py and jobs.py; the known ones get Chinese text.
const JOB_ERROR_TEXT = [
  [/^stylize failed.*timed out after/, () => '风格化超时，这个模型没有风格化版本'],
  [/^stylize failed/, () => '风格化模型生成失败，这个模型没有风格化版本'],
  [/^quality check failed/, () => '生成的模型没有通过质量检查，没有发布'],
  [/^interrupted \d+ times/, () => '服务器在重建这些照片时多次中断，已停止重试'],
  [/no initial image pair/, () => '照片之间重叠太少，找不到可以开始重建的照片'],
  [/only (\d+) of (\d+) photos could be registered/, (m) => `${m[2]} 张照片里只有 ${m[1]} 张能互相对上`],
  [/^need at least \d+ photo/, () => '能用的照片太少'],
];

// Pillow on the server has no HEIF decoder, so HEIC/HEIF uploads come back as unreadable_image.
const HEIF_NAME = /\.(heic|heif)$/i;

export const MODEL_KIND_TEXT = { face: '立面模型', full: '完整模型' };

const FIELD_TEXT = {
  name: '名称',
  lat: '纬度',
  lon: '经度',
  radius_m: '半径',
  id: '标识',
  user_id: '昵称',
  files: '照片',
  landmark_id: '地标',
  kind: '类型',
  sector: '方向',
};

/** Chinese text for a rejection reason or warning code; unknown codes are shown as-is. */
export function codeText(code) {
  return REASON_TEXT[code] ?? WARNING_TEXT[code] ?? String(code);
}

/** Chinese text for why a reconstruction job failed. */
export function jobErrorText(error) {
  const message = String(error ?? '');
  for (const [pattern, text] of JOB_ERROR_TEXT) {
    const m = message.match(pattern);
    if (m) return text(m);
  }
  return '重建程序出错';
}

/** Chinese reasons for one UploadResult, with a format hint for HEIC/HEIF files the server cannot decode. */
export function uploadReasonText(result) {
  return result.reasons
    .map((code) =>
      code === 'unreadable_image' && HEIF_NAME.test(result.filename ?? '')
        ? '暂不支持 HEIC 格式，请导出为 JPEG，或把相机格式设为“兼容性最佳”'
        : codeText(code),
    )
    .join('、');
}

export function isActiveJob(job) {
  return job.status === 'queued' || job.status === 'running';
}

export class ApiError extends Error {
  constructor(status, message, body = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

/** Human-readable message from a FastAPI error body ({detail: str} or {detail: [{loc, msg}]}). */
export function errorMessage(status, body) {
  const detail = body && body.detail;
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d) => {
        const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : '';
        const label = FIELD_TEXT[field] ?? field;
        return label ? `${label}：${d.msg}` : d.msg;
      })
      .join('；');
  }
  return `请求失败（HTTP ${status}）`;
}

async function request(path, { method = 'GET', json, body } = {}) {
  const init = { method, headers: { Accept: 'application/json' } };
  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body;
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError(0, '无法连接服务器，请检查网络后重试');
  }
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      // Proxies and crashed servers answer with HTML; fall through to the status message.
    }
  }
  if (!res.ok) throw new ApiError(res.status, errorMessage(res.status, data), data);
  return data;
}

const enc = encodeURIComponent;

export async function listLandmarks() {
  return (await request('/api/landmarks')).landmarks;
}

export function getLandmark(id) {
  return request(`/api/landmarks/${enc(id)}`);
}

/**
 * The landmark's public photo poses (crowd3d.photo-poses v2, D30): the photos whose uploaders agreed to share where
 * they were taken, rounded by the server; refreshed when a reconstruction job finishes.
 */
export function getPoses(id) {
  return request(`/api/landmarks/${enc(id)}/poses`);
}

/** payload: {name, lat, lon, radius_m?, id?} -> LandmarkSummary */
export function createLandmark(payload) {
  return request('/api/landmarks', { method: 'POST', json: payload });
}

/** -> {results: [UploadResult], triggered_jobs: [JobInfo]} */
/** shareLocation: the uploader ticked "公开这些照片的拍摄位置、朝向和视角" (D30); the field is sent only then. */
export function uploadPhotos({ files, userId, landmarkId = null, shareLocation = false }) {
  const form = new FormData();
  for (const file of files) form.append('files', file, file.name);
  form.append('user_id', userId);
  if (landmarkId) form.append('landmark_id', landmarkId);
  if (shareLocation === true) form.append('share_location', 'true');
  return request('/api/photos', { method: 'POST', body: form });
}

/** Take back the consent to share a photo's location, with the manage token its upload returned (D30). */
export function withdrawShareLocation(photoId, token) {
  return request(`/api/photos/${enc(photoId)}/share_location`, { method: 'POST', json: { share: false, token } });
}

/** kind: "full" | "face" (face needs sector) -> JobInfo */
export function rebuild(landmarkId, kind, sector = null) {
  const json = sector === null || sector === undefined ? { kind } : { kind, sector };
  return request(`/api/landmarks/${enc(landmarkId)}/rebuild`, { method: 'POST', json });
}

export function getJob(jobId) {
  return request(`/api/jobs/${enc(jobId)}`);
}
