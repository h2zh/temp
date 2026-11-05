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

  // web/api.js: the texts for the codes the API returns (the server sends codes and technical English only). Like the
  // Chinese, the texts that end up inside a page's sentence (reasons, warnings, job errors, form errors) end without a
  // full stop; the request errors are whole sentences in English.
  'api.reason.unreadable_image': { zh: '无法读取图片', en: 'The image could not be read' },
  'api.reason.no_exif': {
    zh: '没有 EXIF 信息，请上传相机原图（截图、编辑过或经聊天软件发送的图片会丢失）',
    en: 'No EXIF data: upload the original photo from the camera (screenshots, edited photos and photos sent through chat apps lose it)',
  },
  'api.reason.no_gps': { zh: '缺少GPS定位', en: 'No GPS location' },
  'api.reason.no_landmark': { zh: '附近没有地标', en: 'No landmark nearby' },
  'api.reason.too_far': { zh: '距离地标太远', en: 'Too far from the landmark' },
  'api.reason.facing_away': { zh: '镜头没有朝向建筑', en: 'The camera is not facing the building' },
  'api.reason.too_small': { zh: '分辨率太低', en: 'Resolution too low' },
  'api.reason.too_blurry': { zh: '照片太模糊', en: 'Too blurry' },
  'api.reason.duplicate': { zh: '重复照片', en: 'Duplicate photo' },
  'api.reason.too_large': { zh: '文件过大', en: 'File too large' },
  // "50 米" / "50 m" is Settings.subject_max_distance_m (tests/test_validation.py checks that they agree)
  'api.reason.not_subject': {
    zh: '离建筑超过 50 米，建筑又不是画面主体：请走近拍，或让建筑占满画面',
    en: 'More than 50 m from the building, and the building is not the main subject: move closer, or fill the frame with the building',
  },
  'api.reason.server_error': {
    zh: '服务器没能保存这个文件，其他照片已保留，请稍后重试',
    en: 'The server could not store this file (the other photos are kept): try again later',
  },
  'api.reason.heic': {
    zh: '暂不支持 HEIC 格式，请导出为 JPEG，或把相机格式设为“兼容性最佳”',
    en: 'HEIC photos are not supported yet: export them as JPEG, or set the camera format to "Most Compatible"',
  },
  'api.warning.no_heading': { zh: '缺少指南针朝向', en: 'No compass heading' },
  'api.warning.magnetic_heading': { zh: '使用磁北朝向', en: 'Heading relative to magnetic north' },
  'api.warning.subject_unchecked': { zh: '没有检查建筑是不是照片主体', en: 'Not checked whether the building is the main subject' },
  // between the texts of several codes on one line
  'api.sep.reasons': { zh: '、', en: '. ' },

  'api.jobStatus.queued': { zh: '排队中', en: 'Queued' },
  'api.jobStatus.running': { zh: '生成中', en: 'Running' },
  'api.jobStatus.done': { zh: '已完成', en: 'Done' },
  'api.jobStatus.failed': { zh: '失败', en: 'Failed' },
  'api.jobKind.stylize': { zh: '风格化', en: 'stylization' },
  'api.modelKind.face': { zh: '立面模型', en: 'Facade model' },
  'api.modelKind.full': { zh: '完整模型', en: 'Full model' },

  'api.jobError.stylizeTimeout': { zh: '风格化超时，这个模型没有风格化版本', en: 'Stylizing timed out, so this model has no stylized version' },
  'api.jobError.stylizeFailed': {
    zh: '风格化模型生成失败，这个模型没有风格化版本',
    en: 'The stylized model could not be made, so this model has no stylized version',
  },
  'api.jobError.qualityCheck': { zh: '生成的模型没有通过质量检查，没有发布', en: 'The new model failed the quality check and was not published' },
  'api.jobError.interrupted': {
    zh: '服务器在重建这些照片时多次中断，已停止重试',
    en: 'The server was interrupted several times while reconstructing these photos and has stopped retrying',
  },
  'api.jobError.noInitialPair': {
    zh: '照片之间重叠太少，找不到可以开始重建的照片',
    en: 'The photos overlap too little to find a pair to start the reconstruction from',
  },
  'api.jobError.fewRegistered': {
    zh: '{total} 张照片里只有 {registered} 张能互相对上',
    en: 'Only {registered} of {total} photos could be matched with each other',
  },
  'api.jobError.tooFewPhotos': { zh: '能用的照片太少', en: 'Too few usable photos' },
  'api.jobError.other': { zh: '重建程序出错', en: 'The reconstruction software ran into an error' },

  // a failed request (errorMessage)
  'api.unreachable': { zh: '无法连接服务器，请检查网络后重试', en: 'Could not reach the server. Check your connection and try again.' },
  'api.status.403': { zh: '没有权限进行这项操作', en: "You don't have permission to do this." },
  'api.status.404': { zh: '找不到要访问的内容，可能已被删除', en: 'Not found. It may have been removed.' },
  'api.status.409': { zh: '和服务器上已有的内容冲突', en: 'This conflicts with something already on the server.' },
  'api.status.413': { zh: '上传的内容太大，请少选几张照片后重试', en: 'The upload is too large. Select fewer photos and try again.' },
  'api.status.server': {
    zh: '服务器出错了，请稍后重试（HTTP {status}）',
    en: 'Something went wrong on the server. Try again later (HTTP {status}).',
  },
  'api.status.other': { zh: '请求失败（HTTP {status}）', en: 'The request failed (HTTP {status}).' },
  'api.terms.required': { zh: '请先阅读并同意《用户协议》', en: 'Please read and accept the Terms of Use first.' },
  'api.terms.outdated': {
    zh: '用户协议已更新，请刷新页面后重新阅读并同意',
    en: 'The Terms of Use have changed. Reload the page and accept the new version.',
  },
  // a 422's fields, as the forms call them, and what is wrong with each
  'api.field.name': { zh: '名称', en: 'Name' },
  'api.field.lat': { zh: '纬度', en: 'Latitude' },
  'api.field.lon': { zh: '经度', en: 'Longitude' },
  'api.field.radius_m': { zh: '半径', en: 'Radius' },
  'api.field.id': { zh: '标识', en: 'ID' },
  'api.field.user_id': { zh: '昵称', en: 'Nickname' },
  'api.field.files': { zh: '照片', en: 'Photos' },
  'api.field.landmark_id': { zh: '地标', en: 'Landmark' },
  'api.field.kind': { zh: '类型', en: 'Model type' },
  'api.field.sector': { zh: '方向', en: 'Side' },
  'api.field.terms': { zh: '用户协议', en: 'Terms of Use' },
  'api.invalid.missing': { zh: '{field}：必填', en: '{field}: required' },
  'api.invalid.tooLong': {
    zh: '{field}：最多 {n} 个字符',
    en: { one: '{field}: at most {n} character', other: '{field}: at most {n} characters' },
  },
  'api.invalid.tooShort': {
    zh: '{field}：至少 {n} 个字符',
    en: { one: '{field}: at least {n} character', other: '{field}: at least {n} characters' },
  },
  'api.invalid.min': { zh: '{field}：不能小于 {limit}', en: '{field}: must be at least {limit}' },
  'api.invalid.max': { zh: '{field}：不能大于 {limit}', en: '{field}: must be at most {limit}' },
  'api.invalid.above': { zh: '{field}：必须大于 {limit}', en: '{field}: must be more than {limit}' },
  'api.invalid.value': { zh: '{field}：取值无效', en: '{field}: not a valid value' },
  'api.invalid.request': { zh: '提交的内容有误', en: 'Something in the request is not valid' },
  'api.sep.errors': { zh: '；', en: '; ' },

  // web/models.js
  'models.stylized.pending': { zh: '风格化生成中', en: 'Stylized model in progress' },
  // the stylized model is always clay (D29); whether the photos' registration was accepted
  'models.stylized.aligned': { zh: '素模（照片已对齐）', en: 'Clay model (photos aligned)' },
  'models.stylized.unaligned': { zh: '素模（照片未对齐）', en: 'Clay model (photos not aligned)' },
  'models.stylized.link': { zh: '风格化模型 · {label}', en: 'Stylized model · {label}' },
  // alignment_rmse_m: how far the cameras sit from their own GPS after the alignment
  'models.gpsResidual': { zh: 'GPS 对齐残差', en: 'GPS alignment residual' },

  // the frame of the user agreement page (web/terms.html; the agreement itself is written out in both languages there)
  'terms.docTitle': { zh: '用户协议 · Crowd3D 众拍建模', en: 'Terms of Use · Crowd3D' },
  'terms.metaDescription': {
    zh: '上传照片前需要同意的用户协议：哪些内容会公开、哪些不会、怎样撤回。',
    en: 'The terms you accept before uploading photos: what is made public, what is not, and how to withdraw.',
  },
};
