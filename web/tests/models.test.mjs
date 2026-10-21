// Run from the repo root: node --test 'web/tests/*.test.mjs'
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { modelToShow, sortModels } from '../models.js';

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
