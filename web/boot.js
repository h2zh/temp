// Display preferences every page needs before its first paint (D32): the page language and the colour theme. A
// classic script, loaded synchronously as the first script in <head> of every page (before the stylesheets, the
// import map and the page module), so <html lang>, <html data-lang> and <html data-theme> are right from the first
// frame (no flash of the wrong language or theme). Plain ES2017, no imports or exports; it exposes one global,
// window.crowd3dPrefs.
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

  // ---------- the colour theme ----------
  //
  // Theme choice: 'system' (the default), 'light' or 'dark', kept in localStorage under crowd3d.theme ('system' = no
  // entry; an unknown stored value counts as 'system'). <html data-theme> always holds the resolved 'light' | 'dark',
  // and every dark rule of the pages is keyed on it (style.css, stylized.html, photoposes.js); <html data-theme-choice>
  // holds the choice. 'system' follows prefers-color-scheme live. Unlike the language, a new theme applies at once,
  // without a reload: a choice made here (setTheme), in another tab (storage event) or on another page before Back
  // (pageshow from the back/forward cache), and a change of the system's mode while the choice is 'system'. When
  // storage is blocked the choice lasts as long as this page. Whenever the resolved theme or the choice changes,
  // window gets a 'crowd3d:theme' event (detail {theme, choice}): the switch keeps its value in sync with it. The
  // clay viewer's art themes (画风) never read data-theme: only the panels around the model follow it.
  var THEME_KEY = 'crowd3d.theme';
  var THEME_CHOICES = ['system', 'light', 'dark'];
  var themeMemo = null; // the choice while storage is blocked
  var darkQuery = null;
  try {
    darkQuery = typeof w.matchMedia === 'function' ? w.matchMedia('(prefers-color-scheme: dark)') : null;
  } catch (e) {
    darkQuery = null;
  }

  /** The stored choice: 'system' | 'light' | 'dark'. */
  function themeChoice() {
    if (themeMemo) return themeMemo;
    var v = null;
    try {
      v = w.localStorage.getItem(THEME_KEY);
    } catch (e) {
      /* storage blocked */
    }
    return THEME_CHOICES.indexOf(v) >= 0 ? v : 'system';
  }

  /** 'light' | 'dark' for a choice; 'system' (and anything unknown) takes the system's mode (systemDark). */
  function resolveTheme(choice, systemDark) {
    if (choice === 'light' || choice === 'dark') return choice;
    return systemDark ? 'dark' : 'light';
  }

  /** Puts the stored choice on <html>; tells the page when the theme or the choice changed. */
  function applyTheme() {
    var choice = themeChoice();
    var theme = resolveTheme(choice, !!(darkQuery && darkQuery.matches));
    var before = root.dataset.theme;
    var beforeChoice = root.dataset.themeChoice;
    if (before === theme && beforeChoice === choice) return theme;
    root.dataset.theme = theme;
    root.dataset.themeChoice = choice;
    if (before !== undefined && typeof w.dispatchEvent === 'function' && typeof w.CustomEvent === 'function')
      w.dispatchEvent(new w.CustomEvent('crowd3d:theme', { detail: { theme: theme, choice: choice } }));
    return theme;
  }

  /** Saves a choice ('system' removes the entry) and applies it at once; returns the resolved theme. */
  function setTheme(choice) {
    if (THEME_CHOICES.indexOf(choice) < 0) throw new Error('unknown theme choice: ' + choice);
    try {
      if (choice === 'system') w.localStorage.removeItem(THEME_KEY);
      else w.localStorage.setItem(THEME_KEY, choice);
      themeMemo = null;
    } catch (e) {
      themeMemo = choice; // storage blocked: the choice lasts as long as this page
    }
    return applyTheme();
  }

  applyTheme();
  if (darkQuery) {
    if (typeof darkQuery.addEventListener === 'function') darkQuery.addEventListener('change', applyTheme);
    else if (typeof darkQuery.addListener === 'function') darkQuery.addListener(applyTheme); // Safari before 14
  }
  // a choice made in another tab or window of the site (key null: the storage was cleared)
  w.addEventListener('storage', function (event) {
    if (event.key === THEME_KEY || event.key === null) applyTheme();
  });
  // a page restored from the back/forward cache: the choice may have changed on another page meanwhile
  w.addEventListener('pageshow', function (event) {
    if (event.persisted) applyTheme();
  });

  w.crowd3dPrefs = {
    LANG_KEY: LANG_KEY,
    LANG_CHOICES: LANG_CHOICES,
    langChoice: langChoice,
    resolveLang: resolveLang,
    setLang: setLang,
    THEME_KEY: THEME_KEY,
    THEME_CHOICES: THEME_CHOICES,
    themeChoice: themeChoice,
    resolveTheme: resolveTheme,
    setTheme: setTheme,
  };
})(window);
