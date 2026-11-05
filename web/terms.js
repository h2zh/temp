// The user agreement page (web/terms.html, D32): the page language and the display settings. The agreement itself
// is written out in both languages in the HTML; style.css shows the version of the page language.

import { applyDom } from './i18n.js';
import { mountPrefs } from './ui.js';

applyDom();
mountPrefs(document.getElementById('prefs'));
