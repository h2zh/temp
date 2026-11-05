// What the landmark page says about reconstruction: the next step towards the full model, failed
// builds and how they get retried, and how often to poll. Pure, so node tests cover it. Texts in the page language
// (web/strings.js status.*).

import { isActiveJob, jobErrorText } from './api.js';
import { sectorName } from './compass.js';
import { fmtList, t } from './i18n.js';

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
 * How a failed build gets retried (a clause, without the closing full stop). The planner (crowd3d/coverage.py
 * plan_jobs) retries a target on its own once its photo count exceeds the largest earlier attempt by
 * face/full_rebuild_growth; the exact number is only given when the API sends those thresholds.
 */
export function retryText(d, failed) {
  const th = d.thresholds;
  if (failed.kind === 'face') {
    const growth = th.face_rebuild_growth;
    return Number.isFinite(growth) ? t('status.retry.face', { n: growth }) : t('status.retry.faceSoon');
  }
  const tried = Math.max(...d.jobs.filter((j) => j.kind === 'full').map((j) => j.num_photos));
  const growth = th.full_rebuild_growth;
  // Reconstruction is seeded, so a manual rebuild only helps when photos arrived since the failed attempt.
  const manual = d.num_photos > failed.num_photos;
  if (!Number.isFinite(growth)) return manual ? t('status.retry.fullSoonManual', { total: d.num_photos }) : t('status.retry.fullSoon');
  const params = { n: Math.max(1, tried + growth - d.num_photos), total: d.num_photos };
  return manual ? t('status.retry.fullManual', params) : t('status.retry.full', params);
}

/** The hero's one-line status: what the full model is waiting for. */
export function nextStep(d) {
  const th = d.thresholds;
  if (d.jobs.some((j) => j.kind === 'full' && isActiveJob(j))) return t('status.fullRunning');
  const latest = latestJobs(d.jobs).get('full');
  const failed = latest && latest.status === 'failed' ? latest : null;
  const why = failed ? t('status.why', { error: jobErrorText(failed.error), retry: retryText(d, failed) }) : '';
  if (d.latest_full_model) {
    const n = d.latest_full_model.num_photos;
    return failed ? t('status.fullBuiltUpdateFailed', { n, tried: failed.num_photos, why }) : t('status.fullBuilt', { n });
  }
  if (failed) return t('status.fullFailed', { why });
  if (d.full_ready) return t('status.fullReady');
  const ready = d.sectors.filter((s) => s.state === 'face_ready').length;
  const needSectors = Math.max(0, th.full_min_sectors - ready);
  const needPhotos = Math.max(0, th.full_min_photos - d.num_photos);
  let sides = null;
  if (needSectors) {
    const suggest = d.sectors
      .filter((s) => s.state !== 'face_ready')
      .sort((a, b) => b.count - a.count || a.index - b.index)
      .slice(0, needSectors)
      .map((s) => sectorName(s.label));
    sides = t('status.need.sides', { n: needSectors, min: th.face_min_photos, suggest: fmtList(suggest) });
  }
  const photos = needPhotos ? t('status.need.photos', { n: needPhotos }) : null;
  if (sides && photos) return t('status.need.both', { sides, photos });
  return sides || photos ? t('status.need.one', { need: sides || photos }) : '';
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
