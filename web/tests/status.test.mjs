// Run from the repo root: node --test 'web/tests/*.test.mjs'
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  POLL_QUEUED_MS,
  POLL_RUNNING_MS,
  failedFaceSectors,
  isLatestFailure,
  jobsChanged,
  latestJobs,
  nextStep,
  pollDelay,
  retryText,
} from '../status.js';

const LABELS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const NO_PAIR = 'structure-from-motion found no initial image pair; photos may not overlap enough';

function job(id, kind, sector, status, num_photos, created_at, error = null) {
  return { id, kind, sector, status, num_photos, created_at, error: status === 'failed' ? (error ?? NO_PAIR) : null };
}

function detail({ counts = [5, 5, 5, 5, 5, 5, 0, 0], jobs = [], full = null, thresholds = {} } = {}) {
  const sectors = counts.map((count, index) => ({
    index,
    label: LABELS[index],
    count,
    state: count >= 4 ? 'face_ready' : count ? 'partial' : 'locked',
    bearing_center_deg: index * 45,
    has_face_model: false,
  }));
  const num_photos = counts.reduce((a, b) => a + b, 0);
  return {
    sectors,
    num_photos,
    jobs,
    full_ready: sectors.filter((s) => s.state === 'face_ready').length >= 6 && num_photos >= 24,
    latest_full_model: full,
    thresholds: { face_min_photos: 4, full_min_photos: 24, full_min_sectors: 6, num_sectors: 8, ...thresholds },
  };
}

test('latestJobs keeps the newest job per target', () => {
  const jobs = [job('a', 'face', 0, 'failed', 4, 1), job('b', 'face', 0, 'done', 7, 5), job('c', 'full', null, 'failed', 24, 3)];
  const latest = latestJobs(jobs);
  assert.equal(latest.get('face:0').id, 'b');
  assert.equal(latest.get('full').id, 'c');
  assert.ok(isLatestFailure(jobs, jobs[2]));
  assert.ok(!isLatestFailure(jobs, jobs[0]), 'a later success replaces an old failure');
  assert.deepEqual([...failedFaceSectors(jobs)], []);
});

test('failedFaceSectors: facades whose newest attempt failed, not ones being retried', () => {
  const jobs = [
    job('w1', 'face', 6, 'failed', 4, 1),
    job('n1', 'face', 0, 'failed', 4, 1),
    job('n2', 'face', 0, 'running', 7, 2),
    job('e1', 'face', 2, 'done', 5, 1),
  ];
  assert.deepEqual([...failedFaceSectors(jobs)], [6]);
});

test('hero: a failed full model says so and how it gets retried, instead of "waiting"', () => {
  const failed = job('f', 'full', null, 'failed', 30, 2);
  const d = detail({ jobs: [failed] });
  const text = nextStep(d);
  assert.ok(d.full_ready);
  assert.doesNotMatch(text, /等待生成中/);
  assert.match(text, /^完整模型生成失败：照片之间重叠太少/);
  assert.match(text, /继续上传照片，数量够了会自动重试/);
  assert.doesNotMatch(text, /点“重建”/, 'the same photos would fail again, so no manual rebuild hint');
});

test('hero: retry counts use the growth thresholds when the API sends them', () => {
  const failed = job('f', 'full', null, 'failed', 24, 2);
  const d = detail({ counts: [5, 5, 5, 5, 5, 3, 0, 0], jobs: [job('f0', 'full', null, 'failed', 20, 1), failed], thresholds: { full_rebuild_growth: 8 } });
  assert.equal(d.num_photos, 28);
  // Retried once 24 + 8 = 32 photos are in: 4 more.
  assert.match(nextStep(d), /再上传 4 张照片后会自动重试；也可以点“重建”，用现有的 28 张照片马上重试/);
  const exactlyAt = detail({ counts: [6, 6, 6, 6, 6, 6, 0, 0], jobs: [job('f', 'full', null, 'failed', 24, 2)], thresholds: { full_rebuild_growth: 8 } });
  assert.match(retryText(exactlyAt, exactlyAt.jobs[0]), /再上传 1 张照片后会自动重试/, 'never "0 more"');
});

test('hero: a failed update of an existing full model keeps the model and reports the failure', () => {
  const d = detail({ full: { num_photos: 24 }, jobs: [job('ok', 'full', null, 'done', 24, 1), job('f', 'full', null, 'failed', 30, 2)] });
  assert.match(nextStep(d), /^完整模型已生成（24 张照片参与）。用 30 张照片更新时失败：照片之间重叠太少/);
  const fine = detail({ full: { num_photos: 24 }, jobs: [job('ok', 'full', null, 'done', 24, 1)] });
  assert.equal(nextStep(fine), '完整模型已生成（24 张照片参与）。继续上传可以让模型更完整。');
});

test('hero: an active full job wins over an older failure', () => {
  const d = detail({ jobs: [job('f', 'full', null, 'failed', 24, 1), job('r', 'full', null, 'queued', 32, 2)] });
  assert.equal(nextStep(d), '完整模型正在生成，请稍候…');
});

test('hero: unchanged texts for waiting and for missing sectors/photos', () => {
  assert.equal(nextStep(detail({ jobs: [] })), '照片已足够生成完整模型，等待生成中。');
  const few = detail({ counts: [5, 2, 1, 0, 0, 0, 0, 0] });
  assert.equal(nextStep(few), '距离完整模型：还需 5 个方向各达到 4 张（建议补拍：东北、东、东南、南、西南），总照片还差 16 张。');
});

test('retryText for a facade', () => {
  const failed = job('w', 'face', 6, 'failed', 5, 1);
  assert.equal(retryText(detail(), failed), '这个方向再多几张照片后会自动重试');
  assert.equal(retryText(detail({ thresholds: { face_rebuild_growth: 3 } }), failed), '这个方向的照片比上次多 3 张后会自动重试');
});

test('polling: fast while a job runs, slower while jobs only wait, none when idle', () => {
  assert.equal(pollDelay([job('a', 'face', 0, 'running', 4, 1), job('b', 'face', 1, 'queued', 4, 2)]), POLL_RUNNING_MS);
  assert.equal(pollDelay([job('b', 'face', 1, 'queued', 4, 2), job('c', 'face', 2, 'done', 4, 0)]), POLL_QUEUED_MS);
  assert.equal(pollDelay([job('c', 'face', 2, 'done', 4, 0), job('d', 'face', 3, 'failed', 4, 0)]), null);
  assert.equal(pollDelay([]), null);
});

test('jobsChanged compares polled statuses with the ones on screen', () => {
  const jobs = [job('a', 'face', 0, 'running', 4, 1), job('b', 'face', 1, 'queued', 4, 2)];
  assert.ok(!jobsChanged(jobs, [{ ...jobs[0] }, { ...jobs[1] }]));
  assert.ok(jobsChanged(jobs, [{ ...jobs[0], status: 'done' }]));
  assert.ok(jobsChanged(jobs, [{ ...jobs[1], status: 'running' }]));
  assert.ok(jobsChanged(jobs, [{ id: 'new', status: 'queued' }]));
});

test('stylize jobs build no target: they never hide or count as a failed build', () => {
  const failed = job('f', 'face', 6, 'failed', 4, 10);
  const styl = { ...job('s', 'stylize', 6, 'done', 4, 20), model_id: 'm' };
  const stylFailed = { ...job('t', 'stylize', 2, 'failed', 4, 30, 'stylize failed, model m gets no stylized GLB: x'), model_id: 'm2' };
  const jobs = [failed, styl, stylFailed];
  assert.deepEqual([...latestJobs(jobs).keys()], ['face:6']);
  assert.deepEqual([...failedFaceSectors(jobs)], [6]);
  assert.equal(isLatestFailure(jobs, failed), true);
  assert.equal(isLatestFailure(jobs, stylFailed), false);
  assert.equal(pollDelay([{ ...styl, status: 'running' }]), POLL_RUNNING_MS);   // the page keeps polling meanwhile
});
