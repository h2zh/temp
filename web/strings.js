// Every user-visible text of the web UI in both languages (D32), read through web/i18n.js t(key, params).
// An entry is { zh, en }; an English value that depends on a count is { one, other } (chosen by
// Intl.PluralRules from params.n). Placeholders are {name}; both languages use the same set. The Chinese
// values are the text the pages showed before the switch existed, unchanged; the pages' HTML is still written
// in Chinese and the zh value of a [data-i18n] element equals its text there (web/tests/i18n.test.mjs).
//
// Glossary (zh -> en), so the same thing has the same name everywhere:
//   地标 landmark · 打卡照片 check-in photo · 立面模型 facade model · 完整模型 full model
//   素模（照片已对齐 / 未对齐） Clay model (photos aligned / not aligned) · 共同完成者 Contributors · 昵称 Nickname
//   覆盖进度 Coverage · 方向 side · 公开的拍摄位置 Shared photo positions · 撤回 Withdraw · 用户协议 Terms of Use
//   从这张照片的角度看 View from this photo · 画风 Style · 跟随系统 / 浅色 / 深色 System / Light / Dark · ±范围 ± range
//   Brand: zh 'Crowd3D 众拍建模', en 'Crowd3D'; where the Chinese has the subtitle 众拍建模 on its own, the English
//   one is 'Crowd-built 3D landmarks'.
// English reads as natural English, not word for word; Chinese stays plain Simplified Chinese.

/** The language names in the switch, each in its own language (the same in both page languages). */
export const LANG_NAMES = { zh: '中文', en: 'English' };

export const STRINGS = {
  // the display settings in the page header (web/ui.js mountPrefs)
  'prefs.group': { zh: '显示设置', en: 'Display' },
  'prefs.lang': { zh: '语言', en: 'Language' },
  'prefs.lang.system': { zh: '跟随系统', en: 'System' },

  'brand.sub': { zh: '众拍建模', en: 'Crowd-built 3D landmarks' },
  'common.backToAll': { zh: '← 全部地标', en: '← All landmarks' },

  // web/ui.js formatters
  'unit.m': { zh: '{n} 米', en: '{n} m' },
  'time.justNow': { zh: '刚刚', en: 'just now' },
  'time.minutesAgo': { zh: '{n} 分钟前', en: { one: '{n} minute ago', other: '{n} minutes ago' } },
  'time.hoursAgo': { zh: '{n} 小时前', en: { one: '{n} hour ago', other: '{n} hours ago' } },

  // the frame of the user agreement page (web/terms.html; the agreement itself is written out in both languages there)
  'terms.docTitle': { zh: '用户协议 · Crowd3D 众拍建模', en: 'Terms of Use · Crowd3D' },
  'terms.metaDescription': {
    zh: '上传照片前需要同意的用户协议：哪些内容会公开、哪些不会、怎样撤回。',
    en: 'The terms you accept before uploading photos: what is made public, what is not, and how to withdraw.',
  },
};
