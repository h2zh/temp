// What the landmark page says about reconstruction: the next step towards the full model, failed
// builds and how they get retried, and how often to poll. Pure, so node tests cover it.

import { isActiveJob, jobErrorText } from './api.js';
import { sectorName } from './compass.js';

export const POLL_RUNNING_MS = 3000;
export const POLL_QUEUED_MS = 10000;
export const FULL_REFRESH_MS = 30000; // full details while jobs run, for other visitors' uploads

function targetKey(kind, sector) {
  return kind === 'full' ? 'full' : `face:${sector}`;
}

/** Newest reconstruction job per target ("full" or "face:<sector>"); stylize jobs build no target. */
export function latestJobs(jobs) {
  const latest = new Map();
  for (const j of jobs) {
    if (j.kind !== 'full' && j.kind !== 'face') continue;
    const key = targetKey(j.kind, j.sector);
    const seen = latest.get(key);
    if (!seen || j.created_at > seen.created_at) latest.set(key, j);
  }
  return latest;
}

/** Whether j is the newest job of its target and failed, so nothing for that target is queued or running. */
export function isLatestFailure(jobs, j) {
  return j.status === 'failed' && latestJobs(jobs).get(targetKey(j.kind, j.sector)) === j;
}

/** Sectors whose newest facade job failed. */
export function failedFaceSectors(jobs) {
  const sectors = new Set();
  for (const j of latestJobs(jobs).values()) if (j.kind === 'face' && j.status === 'failed') sectors.add(j.sector);
  return sectors;
}

/**
 * How a failed build gets retried. The planner (crowd3d/coverage.py plan_jobs) retries a target on its own once
 * its photo count exceeds the largest earlier attempt by face/full_rebuild_growth; the exact number is only
 * given when the API sends those thresholds.
 */
export function retryText(d, failed) {
  const t = d.thresholds;
  if (failed.kind === 'face') {
    const growth = t.face_rebuild_growth;
    return Number.isFinite(growth)
      ? `这个方向的照片比上次多 ${growth} 张后会自动重试`
      : '这个方向再多几张照片后会自动重试';
  }
  const tried = Math.max(...d.jobs.filter((j) => j.kind === 'full').map((j) => j.num_photos));
  const growth = t.full_rebuild_growth;
  let text = Number.isFinite(growth)
    ? `再上传 ${Math.max(1, tried + growth - d.num_photos)} 张照片后会自动重试`
    : '继续上传照片，数量够了会自动重试';
  // Reconstruction is seeded, so a manual rebuild only helps when photos arrived since the failed attempt.
  if (d.num_photos > failed.num_photos) text += `；也可以点“重建”，用现有的 ${d.num_photos} 张照片马上重试`;
  return text;
}

/** The hero's one-line status: what the full model is waiting for. */
export function nextStep(d) {
  const t = d.thresholds;
  if (d.jobs.some((j) => j.kind === 'full' && isActiveJob(j))) return '完整模型正在生成，请稍候…';
  const latest = latestJobs(d.jobs).get('full');
  const failed = latest && latest.status === 'failed' ? latest : null;
  const why = failed ? `${jobErrorText(failed.error)}。${retryText(d, failed)}。` : '';
  if (d.latest_full_model) {
    const built = `完整模型已生成（${d.latest_full_model.num_photos} 张照片参与）。`;
    return failed ? `${built}用 ${failed.num_photos} 张照片更新时失败：${why}` : `${built}继续上传可以让模型更完整。`;
  }
  if (failed) return `完整模型生成失败：${why}`;
  if (d.full_ready) return '照片已足够生成完整模型，等待生成中。';
  const ready = d.sectors.filter((s) => s.state === 'face_ready').length;
  const needSectors = Math.max(0, t.full_min_sectors - ready);
  const needPhotos = Math.max(0, t.full_min_photos - d.num_photos);
  const parts = [];
  if (needSectors) {
    const suggest = d.sectors
      .filter((s) => s.state !== 'face_ready')
      .sort((a, b) => b.count - a.count || a.index - b.index)
      .slice(0, needSectors)
      .map((s) => sectorName(s.label));
    parts.push(`还需 ${needSectors} 个方向各达到 ${t.face_min_photos} 张（建议补拍：${suggest.join('、')}）`);
  }
  if (needPhotos) parts.push(`总照片还差 ${needPhotos} 张`);
  return parts.length ? `距离完整模型：${parts.join('，')}。` : '';
}

/** Milliseconds until the next job poll, or null when nothing is queued or running. */
export function pollDelay(jobs) {
  if (jobs.some((j) => j.status === 'running')) return POLL_RUNNING_MS;
  if (jobs.some((j) => j.status === 'queued')) return POLL_QUEUED_MS;
  return null;
}

/** Whether any of the polled jobs has a status other than the one in `jobs`. */
export function jobsChanged(jobs, polled) {
  const before = new Map(jobs.map((j) => [j.id, j.status]));
  return polled.some((j) => before.get(j.id) !== j.status);
}
