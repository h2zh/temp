// Run from the repo root: node --test 'web/tests/*.test.mjs'
// The language choice (D32): web/boot.js, the classic script every page runs first in <head>, run here in a
// node:vm context with a fake window; and the switch web/ui.js mountPrefs draws from it.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { afterEach, test } from 'node:test';
import vm from 'node:vm';

import { mountPrefs } from '../ui.js';
import { inLang, installDom, text } from './fake-dom.mjs';

const SRC = readFileSync(new URL('../boot.js', import.meta.url), 'utf8');

const BLOCKED = {
  getItem() {
    throw new Error('SecurityError');
  },
  setItem() {
    throw new Error('SecurityError');
  },
  removeItem() {
    throw new Error('SecurityError');
  },
};

/** Runs boot.js as a page would, with the browser's languages, what storage holds and the tab's window.name. */
function boot({ languages = ['zh-CN'], language, stored, blocked = false, name = '' } = {}) {
  const classes = new Set();
  const root = {
    lang: '',
    dataset: {},
    classList: { add: (c) => classes.add(c), remove: (c) => classes.delete(c), contains: (c) => classes.has(c) },
  };
  const docListeners = {};
  const winListeners = {};
  const store = new Map(stored === undefined ? [] : [['crowd3d.lang', stored]]);
  let reloads = 0;
  const w = {
    name,
    navigator: { languages, language },
    addEventListener: (type, fn) => (winListeners[type] ??= []).push(fn),
    document: { documentElement: root, addEventListener: (type, fn) => (docListeners[type] ??= []).push(fn) },
    location: { reload: () => reloads++ },
    localStorage: blocked
      ? BLOCKED
      : { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) },
  };
  vm.runInNewContext(SRC, { window: w });
  return {
    w,
    root,
    classes,
    store,
    prefs: w.crowd3dPrefs,
    reloads: () => reloads,
    domContentLoaded: () => (docListeners.DOMContentLoaded ?? []).forEach((fn) => fn()),
    /** The browser showing the page again: persisted = restored whole from the back/forward cache. */
    pageshow: (persisted) => (winListeners.pageshow ?? []).forEach((fn) => fn({ type: 'pageshow', persisted })),
  };
}

test('the system language: the first Chinese or English entry of navigator.languages wins, English without either', () => {
  for (const [languages, lang] of [
    [['zh-TW'], 'zh'],
    [['zh-Hans-CN'], 'zh'],
    [['ZH-cn'], 'zh'],
    [['zh'], 'zh'],
    [['en-GB'], 'en'],
    [['en-GB', 'zh-CN'], 'en'],
    [['ja', 'zh'], 'zh'],
    [['de-DE', 'en-US', 'zh-CN'], 'en'],
    [['zhx', 'enx'], 'en'], // neither: no zh / en subtag
    [['fr'], 'en'],
    [[], 'en'],
  ]) {
    const b = boot({ languages });
    assert.equal(b.root.dataset.lang, lang, JSON.stringify(languages));
    assert.equal(b.prefs.resolveLang('system', languages), lang, JSON.stringify(languages));
    assert.equal(b.root.dataset.langChoice, 'system');
  }
  // a browser without navigator.languages
  assert.equal(boot({ languages: null, language: 'zh-CN' }).root.dataset.lang, 'zh');
  assert.equal(boot({ languages: null }).root.dataset.lang, 'en');
});

test('<html lang>, data-lang and data-lang-choice are set before the first paint', () => {
  const zh = boot({ languages: ['zh-CN'] });
  assert.equal(zh.root.lang, 'zh-CN');
  assert.equal(zh.root.dataset.lang, 'zh');
  assert.equal(zh.root.dataset.langChoice, 'system');
  const en = boot({ languages: ['zh-CN'], stored: 'en' });
  assert.equal(en.root.lang, 'en');
  assert.equal(en.root.dataset.lang, 'en');
  assert.equal(en.root.dataset.langChoice, 'en');
});

test('a stored choice beats the system; a junk stored value counts as the system', () => {
  assert.equal(boot({ languages: ['zh-CN'], stored: 'en' }).root.dataset.lang, 'en');
  assert.equal(boot({ languages: ['en-US'], stored: 'zh' }).root.dataset.lang, 'zh');
  assert.equal(boot({ languages: ['en-US'], stored: 'system' }).root.dataset.lang, 'en');
  for (const junk of ['fr', 'ZH', '', 'null']) {
    const b = boot({ languages: ['zh-CN'], stored: junk });
    assert.equal(b.root.dataset.lang, 'zh', junk);
    assert.equal(b.prefs.langChoice(), 'system', junk);
  }
  assert.equal(boot().prefs.resolveLang('en', ['zh-CN']), 'en');
  assert.equal(boot().prefs.resolveLang('zh', ['en-US']), 'zh');
});

test('setLang saves the choice (system removes it) and reloads; an unknown choice throws and changes nothing', () => {
  const b = boot({ languages: ['zh-CN'] });
  assert.equal(b.prefs.LANG_KEY, 'crowd3d.lang');
  assert.deepEqual([...b.prefs.LANG_CHOICES], ['system', 'zh', 'en']);
  b.prefs.setLang('en');
  assert.equal(b.store.get('crowd3d.lang'), 'en');
  assert.equal(b.prefs.langChoice(), 'en');
  assert.equal(b.reloads(), 1);
  b.prefs.setLang('zh');
  assert.equal(b.store.get('crowd3d.lang'), 'zh');
  b.prefs.setLang('system');
  assert.equal(b.store.has('crowd3d.lang'), false);
  assert.equal(b.prefs.langChoice(), 'system');
  assert.equal(b.reloads(), 3);
  for (const bad of ['fr', 'EN', '', undefined]) assert.throws(() => b.prefs.setLang(bad), /unknown language choice/);
  assert.equal(b.reloads(), 3);
  assert.equal(b.store.has('crowd3d.lang'), false);
});

test('an English page is hidden (html.i18n-pending) until translated; DOMContentLoaded shows it in any case', () => {
  const zh = boot({ languages: ['zh-CN'] });
  assert.equal(zh.classes.has('i18n-pending'), false);
  const en = boot({ languages: ['en-US'] });
  assert.equal(en.classes.has('i18n-pending'), true);
  en.domContentLoaded();
  assert.equal(en.classes.has('i18n-pending'), false);
});

test('blocked storage: the system language, and a choice carried through the reload in window.name', () => {
  const b = boot({ languages: ['zh-CN'], blocked: true });
  assert.equal(b.root.dataset.lang, 'zh');
  assert.equal(b.prefs.langChoice(), 'system');
  b.prefs.setLang('en');
  assert.equal(b.reloads(), 1);
  assert.equal(b.w.name, 'crowd3d.lang=en');
  // the reloaded page
  const again = boot({ languages: ['zh-CN'], blocked: true, name: b.w.name });
  assert.equal(again.root.dataset.lang, 'en');
  assert.equal(again.prefs.langChoice(), 'en');
  again.prefs.setLang('system');
  assert.equal(again.w.name, '');
  // a window.name that is not ours is never overwritten (the choice then cannot outlive the reload)
  const named = boot({ languages: ['zh-CN'], blocked: true, name: 'someone-else' });
  named.prefs.setLang('en');
  assert.equal(named.w.name, 'someone-else');
  assert.equal(named.reloads(), 1);
  // with working storage the stored choice wins, and saving it clears the carried one
  const back = boot({ languages: ['zh-CN'], stored: 'zh', name: 'crowd3d.lang=en' });
  assert.equal(back.root.dataset.lang, 'zh');
  back.prefs.setLang('system');
  assert.equal(back.w.name, '');
  assert.equal(back.root.dataset.lang, 'zh');
});

test('Back / Forward: a page restored from the back/forward cache reloads once when the choice changed meanwhile', () => {
  // the choice changed on another page of the site (storage is shared): restored in the old language -> reload, hidden
  const en = boot({ languages: ['zh-CN'], stored: 'en' });
  en.domContentLoaded();
  en.store.set('crowd3d.lang', 'zh');
  en.pageshow(false); // a fresh load ran boot.js just now: nothing to do
  assert.equal(en.reloads(), 0);
  en.pageshow(true);
  assert.equal(en.reloads(), 1);
  assert.equal(en.classes.has('i18n-pending'), true, 'hidden until the reload, not shown in the old language');
  // nothing changed: no reload, whatever the language
  for (const opts of [{ languages: ['zh-CN'] }, { languages: ['en-US'] }, { languages: ['zh-CN'], stored: 'en' }, { languages: ['en-US'], stored: 'zh' }]) {
    const b = boot(opts);
    b.domContentLoaded();
    b.pageshow(true);
    b.pageshow(true);
    assert.equal(b.reloads(), 0, JSON.stringify(opts));
    assert.equal(b.classes.has('i18n-pending'), false, JSON.stringify(opts));
  }
  // the same language but another choice (system -> zh on a Chinese system): the switch must show the stored choice
  const same = boot({ languages: ['zh-CN'] });
  same.store.set('crowd3d.lang', 'zh');
  same.pageshow(true);
  assert.equal(same.reloads(), 1);
  // back to the system: the entry removed
  const removed = boot({ languages: ['zh-CN'], stored: 'en' });
  removed.store.delete('crowd3d.lang');
  removed.pageshow(true);
  assert.equal(removed.reloads(), 1);
  // following the system, and the browser's language changed while the page was away
  const sys = boot({ languages: ['zh-CN'] });
  sys.w.navigator.languages = ['en-US'];
  sys.pageshow(true);
  assert.equal(sys.reloads(), 1);
  // storage blocked: the choice carried in window.name counts the same
  const blocked = boot({ languages: ['zh-CN'], blocked: true });
  blocked.w.name = 'crowd3d.lang=en';
  blocked.pageshow(true);
  assert.equal(blocked.reloads(), 1);
});

let doc = null;
afterEach(() => {
  delete globalThis.crowd3dPrefs;
  doc?.restore();
  doc = null;
});

function fakePrefs(choice) {
  const calls = [];
  globalThis.crowd3dPrefs = { langChoice: () => choice, setLang: (c) => calls.push(c), themeChoice: () => 'system', setTheme: () => {} };
  return calls;
}

test('the language switch: a labelled select with the current choice; changing it calls setLang', () => {
  doc = installDom();
  const calls = fakePrefs('en');
  const host = doc.getElementById('prefs');
  const group = mountPrefs(host);
  assert.ok(group);
  assert.equal(host.children.length, 1);
  assert.equal(group.className, 'prefs');
  assert.equal(group.getAttribute('role'), 'group');
  assert.equal(group.getAttribute('aria-label'), '显示设置');
  const select = group.querySelector('select');
  const label = group.querySelector('label');
  assert.equal(label.getAttribute('for'), select.getAttribute('id'));
  assert.equal(label.className, 'visually-hidden');
  assert.equal(text(label), '语言');
  assert.equal(group.querySelector('.pref-icon').getAttribute('aria-hidden'), 'true');
  assert.deepEqual(select.options.map((o) => o.getAttribute('value')), ['system', 'zh', 'en']);
  assert.deepEqual(select.options.map((o) => o.textContent), ['跟随系统', '中文', 'English']);
  assert.deepEqual(select.options.map((o) => o.getAttribute('lang')), [null, 'zh-CN', 'en']);
  assert.deepEqual(select.options.filter((o) => o.hasAttribute('selected')).map((o) => o.getAttribute('value')), ['en']);
  assert.equal(select.value, 'en');
  // Back after a switch: a browser restoring form values would show the choice of the page left behind
  assert.equal(select.getAttribute('autocomplete'), 'off');
  select.value = 'zh';
  select.dispatch('change');
  assert.deepEqual(calls, ['zh']);
});

test('the language switch in English names the settings in English and the languages in their own', () => {
  doc = installDom();
  fakePrefs('system');
  const group = inLang('en', () => mountPrefs(doc.getElementById('prefs')));
  assert.equal(group.getAttribute('aria-label'), 'Display');
  assert.equal(text(group.querySelector('label')), 'Language');
  const select = group.querySelector('select');
  assert.deepEqual(select.options.map((o) => o.textContent), ['System', '中文', 'English']);
  assert.equal(select.value, 'system');
});

test('without boot.js (window.crowd3dPrefs) or without a host the switch renders nothing', () => {
  doc = installDom();
  const host = doc.getElementById('prefs');
  assert.equal(mountPrefs(host), null);
  assert.equal(host.children.length, 0);
  fakePrefs('zh');
  assert.equal(mountPrefs(null), null);
});
