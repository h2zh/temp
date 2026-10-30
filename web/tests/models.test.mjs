// Run from the repo root: node --test 'web/tests/*.test.mjs'
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { STYLIZED_PENDING_TEXT, modelToShow, modelsKey, positionSigmaM, sortModels, stylizedView } from '../models.js';

const face = (id, sector, created_at = 1) => ({ id, kind: 'face', sector, created_at });
const full = (id, created_at = 1) => ({ id, kind: 'full', sector: null, created_at });

test('sortModels lists the full model first, then facades by sector', () => {
  const sorted = sortModels([face('e', 2), face('n', 0), full('all'), face('w', 6)]);
  assert.deepEqual(sorted.map((m) => m.id), ['all', 'n', 'e', 'w']);
});

test('sortModels keeps the newest first within a kind and sector, and does not mutate its input', () => {
  const input = [face('old', 0, 1), face('new', 0, 5)];
  assert.deepEqual(sortModels(input).map((m) => m.id), ['new', 'old']);
  assert.deepEqual(input.map((m) => m.id), ['old', 'new']);
});

test('nothing to show without models', () => {
  assert.equal(modelToShow([], null, false), null);
  assert.equal(modelToShow([], face('n', 0), true), null);
});

test('first visit shows the best model', () => {
  assert.equal(modelToShow([full('all'), face('n', 0)], null, false).id, 'all');
  assert.equal(modelToShow([face('n', 0), face('e', 2)], null, false).id, 'n');
});

test('a full model that appears replaces the facade on screen', () => {
  const shown = face('n', 0);
  assert.equal(modelToShow([full('all'), shown], shown, false).id, 'all');
});

test('a facade the visitor picked stays on screen when a full model appears', () => {
  const shown = face('n', 0);
  assert.equal(modelToShow([full('all'), shown], shown, true), null);
});

test('a rebuild of the model on screen replaces it, picked or not', () => {
  const shownFace = face('n1', 0);
  const sortedFaces = [face('n2', 0, 9), face('e', 2)];
  assert.equal(modelToShow(sortedFaces, shownFace, false).id, 'n2');
  assert.equal(modelToShow(sortedFaces, shownFace, true).id, 'n2');
  const shownFull = full('all1');
  assert.equal(modelToShow([full('all2', 9), face('n', 0)], shownFull, true).id, 'all2');
});

test('unchanged models or a new facade elsewhere keep the view', () => {
  const shown = face('n', 0);
  assert.equal(modelToShow([shown, face('e', 2)], shown, false), null);
  const shownFull = full('all');
  assert.equal(modelToShow([shownFull, face('n', 0)], shownFull, false), null);
});

const READY = (accepted, label) => ({
  status: 'ready',
  url: '/files/models/lm/m1.glb',
  registration: { accepted, reason: accepted ? null : 'ambiguous' },
  label,
});

test('stylizedView: nothing without a stylized version, pending text while one is made', () => {
  assert.equal(stylizedView(face('n', 0)), null);
  assert.equal(stylizedView({ ...face('n', 0), stylized: null }), null);
  assert.deepEqual(stylizedView({ ...face('n', 0), stylized: { status: 'pending', url: null } }), {
    pending: true,
    text: STYLIZED_PENDING_TEXT,
  });
});

test('stylizedView: a ready GLB links the stylized viewer with the server label; all clay says so', () => {
  const clay = stylizedView({ ...face('n', 0), stylized: READY(false, '素模（照片未对齐）') });
  assert.deepEqual(clay, {
    href: 'stylized.html?model=%2Ffiles%2Fmodels%2Flm%2Fm1.glb',
    text: '风格化模型 · 素模（照片未对齐）',
    clay: true,
  });
  const aligned = stylizedView({ ...face('n', 0), stylized: READY(true, '风格化（照片已对齐）') });
  assert.equal(aligned.clay, false);
  assert.equal(aligned.text, '风格化模型 · 风格化（照片已对齐）');
  // no label from an older server: the registration decides
  assert.equal(stylizedView({ ...face('n', 0), stylized: READY(false, null) }).text, '风格化模型 · 素模（照片未对齐）');
});

test('modelsKey changes when a model gets its stylized version, not on every poll', () => {
  const before = modelsKey([face('n', 0)]);
  assert.equal(modelsKey([face('n', 0)]), before);
  const pending = modelsKey([{ ...face('n', 0), stylized: { status: 'pending' } }]);
  const ready = modelsKey([{ ...face('n', 0), stylized: READY(false, '素模（照片未对齐）') }]);
  assert.equal(new Set([before, pending, ready]).size, 3);
});

test('positionSigmaM: only for a fully georegistered model whose anchor calibration passed the held-out check', () => {
  const geo = (over) => ({
    ...face('n', 0),
    geo: { level: 'full', calibrated: true, position_sigma_m: { horizontal: 4.2, vertical: 1.6 }, ...over },
  });
  assert.equal(positionSigmaM(geo({})), 4.2);
  assert.equal(positionSigmaM(geo({ calibrated: false })), null); // the default until the final evaluation
  assert.equal(positionSigmaM(geo({ level: 'prior_only' })), null);
  assert.equal(positionSigmaM(geo({ position_sigma_m: { horizontal: null, vertical: null } })), null);
  assert.equal(positionSigmaM({ ...face('n', 0), geo: null }), null); // a model recorded before anchors
  assert.equal(positionSigmaM(face('n', 0)), null); // an older server
});
