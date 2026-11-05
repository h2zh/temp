// Display preferences every page needs before its first paint (D32): the page language. A classic script,
// loaded synchronously as the first script in <head> of every page (before the stylesheets, the import map and
// the page module), so <html lang> and <html data-lang> are right from the first frame. Plain ES2017, no imports
// or exports; it exposes one global, window.crowd3dPrefs, which the colour theme will extend.
//
// Language choice: 'system' (the default), 'zh' or 'en', kept in localStorage under crowd3d.lang ('system' = no
// entry; an unknown stored value counts as 'system'). 'system' follows the browser: the first entry of
// navigator.languages that is Chinese (zh, zh-*) or English (en, en-*) wins, and a browser with neither gets
// English. The pages are written in Chinese; web/i18n.js translates them from <html data-lang>. Switching the
// language saves the choice and reloads the page (nothing re-renders live, and the language does not follow a
// change of the system language while a page is open). When storage is blocked, the choice is kept in this tab's
// window.name instead: it survives the reload and lasts as long as the tab.
//
// An English page starts with <html class="i18n-pending"> (the page's CSS hides the body) until the page module has
// translated it (i18n.js applyDom); DOMContentLoaded removes the class too, in case the module never runs.
//
// Back and Forward: a page the browser restores whole from its back/forward cache (pageshow with persisted) is in the
// language it was left in; when the choice was changed on another page meanwhile (or, following the system, the
// browser's language), it is reloaded (hidden until then), so the page and its switch show the stored choice.
(function (w) {
  'use strict';
  var LANG_KEY = 'crowd3d.lang';
  var LANG_CHOICES = ['system', 'zh', 'en'];
  var NAME_PREFIX = LANG_KEY + '='; // window.name while storage is blocked: 'crowd3d.lang=en'
  var root = w.document.documentElement;

  function known(v) {
    return LANG_CHOICES.indexOf(v) >= 0;
  }

  function nameChoice() {
    try {
      var name = String(w.name || '');
      var v = name.indexOf(NAME_PREFIX) === 0 ? name.slice(NAME_PREFIX.length) : null;
      return known(v) ? v : null;
    } catch (e) {
      return null;
    }
  }

  /** The stored choice: 'system' | 'zh' | 'en'. */
  function langChoice() {
    var v = null;
    try {
      v = w.localStorage.getItem(LANG_KEY);
    } catch (e) {
      /* storage blocked */
    }
    return known(v) ? v : nameChoice() || 'system';
  }

  function setName(value) {
    try {
      var name = String(w.name || '');
      if (!name || name.indexOf(NAME_PREFIX) === 0) w.name = value; // never overwrite a name that is not ours
    } catch (e) {
      /* nowhere to keep it: the page follows the system again after the reload */
    }
  }

  /** 'zh' | 'en' for a choice; 'system' looks at the browser's languages (navigator.languages). */
  function resolveLang(choice, languages) {
    if (choice === 'zh' || choice === 'en') return choice;
    var list = languages || [];
    for (var i = 0; i < list.length; i++) {
      var tag = String(list[i] || '');
      if (/^zh(-|$)/i.test(tag)) return 'zh';
      if (/^en(-|$)/i.test(tag)) return 'en';
    }
    return 'en';
  }

  function systemLanguages() {
    var nav = w.navigator || {};
    if (nav.languages && nav.languages.length) return nav.languages;
    return nav.language ? [nav.language] : [];
  }

  /** Saves a choice ('system' removes the entry) and reloads the page in that language. */
  function setLang(choice) {
    if (!known(choice)) throw new Error('unknown language choice: ' + choice);
    try {
      if (choice === 'system') w.localStorage.removeItem(LANG_KEY);
      else w.localStorage.setItem(LANG_KEY, choice);
      setName('');
    } catch (e) {
      // storage blocked: carry the choice through the reload in this tab's window.name
      setName(choice === 'system' ? '' : NAME_PREFIX + choice);
    }
    w.location.reload();
  }

  var choice = langChoice();
  var lang = resolveLang(choice, systemLanguages());
  root.lang = lang === 'zh' ? 'zh-CN' : 'en';
  root.dataset.lang = lang;
  root.dataset.langChoice = choice;
  if (lang === 'en') {
    root.classList.add('i18n-pending');
    w.document.addEventListener('DOMContentLoaded', function () {
      root.classList.remove('i18n-pending');
    });
  }

  // a page restored from the back/forward cache whose language choice changed while it was away
  w.addEventListener('pageshow', function (event) {
    if (!event.persisted) return;
    var now = langChoice();
    if (now === root.dataset.langChoice && resolveLang(now, systemLanguages()) === root.dataset.lang) return;
    root.classList.add('i18n-pending');
    w.location.reload();
  });

  w.crowd3dPrefs = {
    LANG_KEY: LANG_KEY,
    LANG_CHOICES: LANG_CHOICES,
    langChoice: langChoice,
    resolveLang: resolveLang,
    setLang: setLang,
  };
})(window);
