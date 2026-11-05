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
//   立面起伏 relief (a feature of it) · 倾斜 lean · 可撤销 can be undone
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
  'prefs.theme': { zh: '外观', en: 'Appearance' },
  'prefs.theme.system': { zh: '跟随系统', en: 'System' },
  'prefs.theme.light': { zh: '浅色', en: 'Light' },
  'prefs.theme.dark': { zh: '深色', en: 'Dark' },

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

  // texts several pages share
  'common.loading': { zh: '正在加载…', en: 'Loading…' },
  'common.retry': { zh: '重试', en: 'Try again' },
  'common.terms': { zh: '用户协议', en: 'Terms of Use' },
  'common.coverage': { zh: '覆盖进度', en: 'Coverage' },
  'common.coverageValue': { zh: '覆盖进度 {percent}', en: 'Coverage {percent}' },
  'common.photoCount': { zh: '{n} 张照片', en: { one: '{n} photo', other: '{n} photos' } },
  'common.contributorCount': { zh: '{n} 位参与者', en: { one: '{n} contributor', other: '{n} contributors' } },
  // a clause made a sentence, and between whole sentences (Chinese sentences end with 。 and need no space)
  'common.sentence': { zh: '{text}。', en: '{text}.' },
  'common.sentenceSep': { zh: '', en: ' ' },

  // web/index.html and web/index.js: the home page
  'home.docTitle': { zh: 'Crowd3D 众拍建模', en: 'Crowd3D' },
  'home.metaDescription': {
    zh: '在地标周围不同方向打卡拍照，大家的照片一起生成建筑的 3D 模型。',
    en: "Take check-in photos of a landmark from every side: together, everyone's photos build a 3D model of it.",
  },
  'home.title': { zh: '一起把地标拍成 3D', en: 'Build landmarks in 3D, together' },
  'home.intro': {
    zh: '站在地标周围的不同方向打卡拍照。某个方向的照片够了，就先生成这一面的立面模型；大部分方向都有人拍过之后，生成完整的 3D 模型。',
    en: 'Take check-in photos from different sides of a landmark. As soon as one side has enough photos, a facade model of that side is built; once most sides have been photographed, the full 3D model follows.',
  },
  'home.map': { zh: '地图', en: 'Map' },
  'home.mapLabel': { zh: '地标地图', en: 'Landmark map' },
  'home.mapHint': { zh: '点击地图可把坐标填入下方的“新建地标”。', en: 'Click the map to put its coordinates into "New landmark" below.' },
  'home.map.loading': { zh: '地图加载中…', en: 'Loading the map…' },
  'home.map.slow': { zh: '地图加载较慢，可以先使用地标列表。', en: 'The map is slow to load. You can use the landmark list meanwhile.' },
  'home.map.failed': { zh: '地图组件加载失败，可直接使用下方列表。', en: 'The map could not be loaded. You can still use the list below.' },
  // the zoom buttons of the map (Leaflet's own titles are English)
  'map.zoomIn': { zh: '放大', en: 'Zoom in' },
  'map.zoomOut': { zh: '缩小', en: 'Zoom out' },
  'home.list': { zh: '地标', en: 'Landmarks' },
  'home.list.count': { zh: '共 {n} 个', en: { one: '{n} landmark', other: '{n} landmarks' } },
  'home.list.empty': { zh: '还没有地标。在“新建地标”里添加第一个吧。', en: 'No landmarks yet. Add the first one under "New landmark".' },
  'home.list.loadFailed': { zh: '地标列表加载失败：{message}', en: 'The landmark list could not be loaded: {message}' },
  'home.status.full': { zh: '完整模型已生成', en: 'Full model built' },
  'home.status.fullReady': { zh: '可生成完整模型', en: 'Ready for the full model' },
  'home.status.faces': { zh: '{n} 个立面已建模', en: { one: '{n} facade built', other: '{n} facades built' } },
  'home.status.collecting': { zh: '收集照片中', en: 'Collecting photos' },
  'home.status.waiting': { zh: '等待第一张照片', en: 'Waiting for the first photo' },
  'home.card.progressLabel': { zh: '{name} 覆盖进度', en: 'Coverage of {name}' },
  'home.details': { zh: '查看详情 →', en: 'View details →' },
  'home.new': { zh: '新建地标', en: 'New landmark' },
  'home.form.name': { zh: '名称', en: 'Name' },
  'home.form.namePlaceholder': { zh: '例如：外滩钟楼', en: 'e.g. Bund Clock Tower' },
  'home.form.lat': { zh: '纬度', en: 'Latitude' },
  'home.form.lon': { zh: '经度', en: 'Longitude' },
  'home.form.radius': { zh: '打卡半径（米）', en: 'Check-in radius (m)' },
  'home.form.useLocation': { zh: '使用当前位置', en: 'Use my location' },
  'home.form.create': { zh: '创建', en: 'Create' },
  'home.create.nameMissing': { zh: '请填写名称', en: 'Please enter a name.' },
  'home.create.creating': { zh: '正在创建…', en: 'Creating…' },
  'home.create.done': { zh: '已创建「{name}」。', en: 'Created "{name}".' },
  'home.create.upload': { zh: '去上传照片 →', en: 'Upload photos →' },
  'home.create.duplicate': {
    zh: '已存在相同标识的地标，请换一个名称。',
    en: 'A landmark with the same ID already exists. Choose another name.',
  },
  'home.create.invalid': { zh: '填写内容有误：{message}', en: 'Please check the form: {message}' },
  'home.create.failed': { zh: '创建失败：{message}', en: 'Could not create the landmark: {message}' },
  'home.locate.unsupported': { zh: '当前浏览器不支持定位', en: 'This browser cannot tell your location.' },
  'home.locate.locating': { zh: '正在定位…', en: 'Finding your location…' },
  'home.locate.done': { zh: '已填入当前位置', en: 'Your location has been filled in.' },
  'home.locate.failed': {
    zh: '无法获取当前位置，请在地图上点击选择',
    en: 'Could not get your location. Click the map to pick a point instead.',
  },

  // web/compass.js: the coverage compass. A sector's name inside a sentence (lower case in English), its short name
  // around the rose, and its facade ({side} is the name at the start of a line: North-east)
  'compass.sector.N': { zh: '北', en: 'north' },
  'compass.sector.NE': { zh: '东北', en: 'north-east' },
  'compass.sector.E': { zh: '东', en: 'east' },
  'compass.sector.SE': { zh: '东南', en: 'south-east' },
  'compass.sector.S': { zh: '南', en: 'south' },
  'compass.sector.SW': { zh: '西南', en: 'south-west' },
  'compass.sector.W': { zh: '西', en: 'west' },
  'compass.sector.NW': { zh: '西北', en: 'north-west' },
  'compass.short.N': { zh: '北', en: 'N' },
  'compass.short.NE': { zh: '东北', en: 'NE' },
  'compass.short.E': { zh: '东', en: 'E' },
  'compass.short.SE': { zh: '东南', en: 'SE' },
  'compass.short.S': { zh: '南', en: 'S' },
  'compass.short.SW': { zh: '西南', en: 'SW' },
  'compass.short.W': { zh: '西', en: 'W' },
  'compass.short.NW': { zh: '西北', en: 'NW' },
  'compass.face': { zh: '{side}立面', en: '{side} face' },
  'compass.state.locked': { zh: '暂无照片', en: 'No photos yet' },
  'compass.state.partial': { zh: '照片不足', en: 'Not enough photos' },
  'compass.state.face_ready': { zh: '照片已够', en: 'Enough photos' },
  'compass.faceBuilt': { zh: '已生成立面', en: 'Facade built' },
  'compass.faceFailed': { zh: '立面生成失败', en: 'Facade build failed' },
  'compass.empty': { zh: '暂无覆盖数据', en: 'No coverage data yet' },
  'compass.summary': {
    zh: '覆盖罗盘：{sides} 个方向中 {ready} 个照片已够，共 {n} 张照片',
    en: {
      one: 'Coverage compass: enough photos on {ready} of {sides} sides, {n} photo in all',
      other: 'Coverage compass: enough photos on {ready} of {sides} sides, {n} photos in all',
    },
  },
  // a wedge's tooltip: the sector, then its details one after the other
  'compass.tip': { zh: '{side}（{label}）：{details}', en: '{side} ({label}): {details}.' },
  'compass.tip.sep': { zh: '，', en: '. ' },
  'compass.tip.count': { zh: '{n} 张', en: { one: '{n} photo', other: '{n} photos' } },
  'compass.tip.countOf': { zh: '{n} 张 / {min}', en: { one: '{n} photo ({min} needed)', other: '{n} photos ({min} needed)' } },
  'compass.tip.built': { zh: '已生成立面模型', en: 'Facade model built' },
  'compass.tip.updateFailed': { zh: '最近一次更新失败', en: 'Latest update failed' },
  // under the total in the middle of the rose
  'compass.totalUnit': { zh: '张', en: { one: 'photo', other: 'photos' } },

  // web/status.js: the next step towards the full model, and how a failed build gets retried (clauses without the
  // closing full stop)
  'status.retry.face': {
    zh: '这个方向的照片比上次多 {n} 张后会自动重试',
    en: {
      one: 'It will be retried automatically once this side has {n} more photo than last time',
      other: 'It will be retried automatically once this side has {n} more photos than last time',
    },
  },
  'status.retry.faceSoon': { zh: '这个方向再多几张照片后会自动重试', en: 'It will be retried automatically once this side has a few more photos' },
  'status.retry.full': {
    zh: '再上传 {n} 张照片后会自动重试',
    en: { one: 'It will be retried automatically after {n} more photo', other: 'It will be retried automatically after {n} more photos' },
  },
  'status.retry.fullManual': {
    zh: '再上传 {n} 张照片后会自动重试；也可以点“重建”，用现有的 {total} 张照片马上重试',
    en: {
      one: 'It will be retried automatically after {n} more photo; or press "Rebuild" to retry now with the {total} photos already here',
      other: 'It will be retried automatically after {n} more photos; or press "Rebuild" to retry now with the {total} photos already here',
    },
  },
  'status.retry.fullSoon': { zh: '继续上传照片，数量够了会自动重试', en: 'Keep uploading photos: it will be retried automatically once there are enough' },
  'status.retry.fullSoonManual': {
    zh: '继续上传照片，数量够了会自动重试；也可以点“重建”，用现有的 {total} 张照片马上重试',
    en: 'Keep uploading photos: it will be retried automatically once there are enough; or press "Rebuild" to retry now with the {total} photos already here',
  },
  // why a build failed (api.jobError.*) and how it gets retried, as two sentences
  'status.why': { zh: '{error}。{retry}。', en: '{error}. {retry}.' },
  'status.fullRunning': { zh: '完整模型正在生成，请稍候…', en: 'The full model is being built. Please wait…' },
  'status.fullBuilt': {
    zh: '完整模型已生成（{n} 张照片参与）。继续上传可以让模型更完整。',
    en: {
      one: 'The full model is built from {n} photo. More uploads make it more complete.',
      other: 'The full model is built from {n} photos. More uploads make it more complete.',
    },
  },
  'status.fullBuiltUpdateFailed': {
    zh: '完整模型已生成（{n} 张照片参与）。用 {tried} 张照片更新时失败：{why}',
    en: {
      one: 'The full model is built from {n} photo. Updating it with {tried} photos failed: {why}',
      other: 'The full model is built from {n} photos. Updating it with {tried} photos failed: {why}',
    },
  },
  'status.fullFailed': { zh: '完整模型生成失败：{why}', en: 'The full model could not be built: {why}' },
  'status.fullReady': { zh: '照片已足够生成完整模型，等待生成中。', en: 'There are enough photos for the full model, which is waiting to be built.' },
  'status.need.sides': {
    zh: '还需 {n} 个方向各达到 {min} 张（建议补拍：{suggest}）',
    en: {
      one: '{n} more side with {min} photos (suggested: {suggest})',
      other: '{n} more sides with {min} photos each (suggested: {suggest})',
    },
  },
  'status.need.photos': { zh: '总照片还差 {n} 张', en: { one: '{n} more photo in all', other: '{n} more photos in all' } },
  'status.need.both': { zh: '距离完整模型：{sides}，{photos}。', en: 'Still needed for the full model: {sides}, and {photos}.' },
  'status.need.one': { zh: '距离完整模型：{need}。', en: 'Still needed for the full model: {need}.' },

  // web/landmark.html and web/landmark.js: the landmark page
  'landmark.pageTitle': { zh: '地标 · Crowd3D 众拍建模', en: 'Landmark · Crowd3D' },
  'landmark.docTitle': { zh: '{name} · Crowd3D 众拍建模', en: '{name} · Crowd3D' },
  'landmark.metaDescription': {
    zh: '地标各个方向的照片覆盖情况、上传打卡照片、查看生成的 3D 模型。',
    en: "A landmark's photo coverage on every side: upload check-in photos and view the 3D models built from them.",
  },
  'landmark.loading': { zh: '加载中…', en: 'Loading…' },
  'landmark.radius': { zh: '打卡半径 {n} 米', en: 'Check-in radius {n} m' },
  'landmark.cannotOpen': { zh: '无法打开地标', en: 'Could not open the landmark' },
  'landmark.backToList': { zh: '返回地标列表', en: 'Back to the landmark list' },
  'landmark.notFound': { zh: '找不到这个地标。', en: 'There is no such landmark.' },
  'landmark.noId': { zh: '链接里缺少地标 id。', en: 'The link does not say which landmark to show.' },
  'landmark.loadFailed': { zh: '加载失败：{message}', en: 'Could not load the landmark: {message}' },
  'landmark.refreshFailed': { zh: '刷新失败：{message}', en: 'Could not refresh: {message}' },
  // a sector the details do not list, by its index
  'landmark.sectorFallback': { zh: '方向 {n}', en: 'Side {n}' },
  'landmark.coverageTitle': { zh: '各方向覆盖', en: 'Coverage by side' },
  'landmark.legend': { zh: '图例', en: 'Legend' },
  'landmark.coverageRule': {
    zh: '照片按拍摄者站的方向分成 {sectors} 份。每个方向满 {faceMin} 张生成该方向的立面模型；{fullSectors} 个方向达标且共 {fullPhotos} 张后生成完整模型。',
    en: 'Photos are grouped into {sectors} sides by where the photographer stood. A side with {faceMin} photos gets its own facade model; once {fullSectors} sides have enough and there are {fullPhotos} photos in all, the full model is built.',
  },
  'landmark.modelTitle': { zh: '3D 模型', en: '3D models' },
  'landmark.noModel': {
    zh: '还没有模型。任一方向的照片够了，就会自动生成这一面的立面模型。',
    en: 'No model yet. As soon as any side has enough photos, a facade model of that side is built automatically.',
  },
  'landmark.viewerHelp': {
    zh: '拖动旋转 · 滚轮或双指缩放 · 右键或双指平移',
    en: 'Drag to rotate · Scroll or pinch to zoom · Right-drag or two-finger drag to pan',
  },
  'landmark.showPoses': { zh: '显示公开的拍摄位置', en: 'Show shared photo positions' },
  'landmark.modelList': { zh: '模型列表', en: 'Models' },
  // a model's line in the list (the GPS residual's label is models.gpsResidual)
  'models.points': { zh: '{n} 个点', en: { one: '{n} point', other: '{n} points' } },
  'models.registered': { zh: '{registered}/{n} 张照片定位成功', en: { one: '{registered} of {n} photo located', other: '{registered} of {n} photos located' } },
  'models.geo': { zh: '已对齐地理坐标', en: 'Georeferenced' },
  'models.geoResidual': { zh: '已对齐地理坐标（{label} {value}）', en: 'Georeferenced ({label} {value})' },
  'models.positionSigma': { zh: '位置 ±{value}（1σ）', en: 'Position ±{value} (1σ)' },
  'models.notGeo': { zh: '未对齐地理坐标', en: 'Not georeferenced' },
  'models.headingError': { zh: '朝向误差 {value}°', en: 'Heading error {value}°' },

  // web/viewer.js and the landmark page's viewer
  'viewer.label': { zh: '三维点云模型，可拖动旋转、缩放', en: '3D point cloud model: drag to rotate, scroll or pinch to zoom' },
  'viewer.north': { zh: '北', en: 'N' },
  'viewer.noPoints': { zh: '模型中没有点', en: 'The model has no points' },
  'viewer.loading': { zh: '正在加载模型…', en: 'Loading the model…' },
  'viewer.loadFailed': { zh: '模型加载失败：{message}', en: 'The model could not be loaded: {message}' },
  'viewer.unavailable': {
    zh: '3D 查看器无法启动（浏览器可能不支持 WebGL，或网络无法加载 three.js）。',
    en: 'The 3D viewer could not start (the browser may not support WebGL, or three.js could not be loaded).',
  },

  // the manual rebuild
  'rebuild.label': { zh: '手动重建', en: 'Manual rebuild' },
  'rebuild.button': { zh: '重建', en: 'Rebuild' },
  'rebuild.none': { zh: '照片不足，暂不能重建', en: 'Not enough photos to rebuild yet' },
  'rebuild.submitting': { zh: '正在提交…', en: 'Submitting…' },
  'rebuild.queued': { zh: '已加入队列：{title}', en: 'Queued: {title}' },
  'rebuild.notEnough': { zh: '照片数量还不够，暂时不能重建。', en: 'There are not enough photos to rebuild yet.' },
  'rebuild.failed': { zh: '重建失败：{message}', en: 'The rebuild could not be started: {message}' },

  // the jobs
  'jobs.title': { zh: '重建任务', en: 'Reconstruction jobs' },
  'jobs.polling': { zh: '自动刷新中', en: 'Refreshing automatically' },
  'jobs.empty': { zh: '暂无任务。某个方向的照片够了会自动开始。', en: 'No jobs yet. One starts automatically once a side has enough photos.' },
  // a stylize job: the model it works on, and the job kind (api.jobKind.stylize)
  'jobs.stylizeTitle': { zh: '{title}（{kind}）', en: '{title} ({kind})' },
  'jobs.queuedAt': { zh: '{time}加入队列', en: 'Queued {time}' },
  'jobs.startedAt': { zh: '{time}开始', en: 'Started {time}' },
  'jobs.doneAt': { zh: '{time}完成', en: 'Finished {time}' },
  'jobs.doneAtTook': { zh: '{time}完成，用时 {n} 秒', en: { one: 'Finished {time}, took {n} second', other: 'Finished {time}, took {n} seconds' } },
  'jobs.failedAt': { zh: '{time}失败', en: 'Failed {time}' },
  // above a failed job's raw error (technical English, not translated)
  'jobs.technical': { zh: '技术信息', en: 'Technical details' },

  // the contributors
  'contributors.title': { zh: '共同完成者', en: 'Contributors' },
  'contributors.listLabel': { zh: '共同完成者（按昵称排序）', en: 'Contributors (sorted by nickname)' },
  'contributors.empty': { zh: '还没有人上传，成为第一个吧。', en: 'No uploads yet. Be the first!' },
  'contributors.me': { zh: '{name}（你）', en: '{name} (you)' },

  // the upload form, its agreement box (D32: the hint keeps every statement of the Chinese) and its results
  'upload.title': { zh: '上传打卡照片', en: 'Upload check-in photos' },
  'upload.nickname': { zh: '你的昵称', en: 'Your nickname' },
  'upload.nicknamePlaceholder': { zh: '会显示在共同完成者里', en: 'Shown in the contributors list' },
  'upload.files': { zh: '照片（一次最多 50 张）', en: 'Photos (up to 50 at a time)' },
  'upload.hint': {
    zh: '请上传手机相机拍摄的原图（保留 GPS 和指南针信息），站在建筑外围、镜头对准建筑。',
    en: "Upload the original photos from your phone's camera (with their GPS and compass data), taken from around the building with the camera pointing at it.",
  },
  // the label of the agreement box: before the link, the link, after it
  'upload.terms.pre': { zh: '我已阅读并同意', en: 'I have read and accept the ' },
  'upload.terms.link': { zh: '《用户协议》', en: 'Terms of Use' },
  'upload.terms.post': { zh: '', en: '.' },
  'upload.terms.hint': {
    zh: '上传的照片会公开显示（去掉 EXIF 等拍摄信息，缩小到长边不超过 1600 像素），连同拍摄位置（取整到 1 米）、朝向（取整到 1°）和视角（取整到 5°）。你的昵称会出现在共同完成者里，但不显示哪张照片是谁传的、什么时候传的。公开的照片和位置要等这里下一次建模结束后才显示，但任何人都能用手动重建或者上传几张照片让下一次建模很快开始，所以别人可能把那时新出现的照片和你的昵称对上，猜出哪些是你拍的；这里上传的人少时更容易。不同意就不能上传；上传后可以在这台设备上撤回。',
    en: 'Your photos are shown publicly (with EXIF and other capture data removed, scaled down to at most 1600 px on the longer side), together with where they were taken (rounded to 1 m), their heading (rounded to 1°) and field of view (rounded to 5°). Your nickname appears among the contributors, but not which photo anyone uploaded or when. Public photos and positions only appear after the next build here has finished, but anyone can make that build start soon with a manual rebuild or by uploading a few photos, so someone could match the photos that appear then with your nickname and guess which ones are yours; this is easier when few people upload here. If you do not accept, you cannot upload; after uploading, you can withdraw your photos on this device.',
  },
  'upload.submit': { zh: '上传', en: 'Upload' },
  'upload.submitting': { zh: '上传中…', en: 'Uploading…' },
  'upload.nicknameMissing': { zh: '请填写昵称（不超过 64 个字符）', en: 'Please enter a nickname (up to 64 characters).' },
  'upload.filesMissing': { zh: '请选择要上传的照片', en: 'Please choose the photos to upload.' },
  'upload.tooMany': { zh: '一次最多上传 {max} 张，你选了 {n} 张', en: 'You can upload up to {max} photos at a time; you chose {n}.' },
  'upload.checking': {
    zh: '正在上传并检查 {n} 张照片…',
    en: { one: 'Uploading and checking {n} photo…', other: 'Uploading and checking {n} photos…' },
  },
  'upload.failed': { zh: '上传失败：{message}', en: 'Upload failed: {message}' },
  'upload.summary': { zh: '{n} 张通过。', en: { one: '{n} photo accepted.', other: '{n} photos accepted.' } },
  'upload.summaryRejected': {
    zh: '{n} 张通过，{rejected} 张未通过。',
    en: { one: '{n} photo accepted, {rejected} rejected.', other: '{n} photos accepted, {rejected} rejected.' },
  },
  // after the summary: the builds the upload started (facade names or the full model)
  'upload.triggered': { zh: '已开始生成：{list}。', en: 'Now building: {list}.' },
  // its items, inside the sentence ({side}: a sector's prose name, 北 / north)
  'upload.triggered.face': { zh: '{side}立面', en: 'the {side} face' },
  'upload.triggered.full': { zh: '完整模型', en: 'the full model' },
  'upload.results': { zh: '上传结果', en: 'Upload results' },
  'upload.col.photo': { zh: '照片', en: 'Photo' },
  'upload.col.result': { zh: '结果', en: 'Result' },
  'upload.accepted': { zh: '通过', en: 'Accepted' },
  'upload.rejected': { zh: '未通过', en: 'Rejected' },
  'upload.public': { zh: '已公开（可撤回）', en: 'Public (you can withdraw it)' },
  'upload.notPublic': { zh: '未公开', en: 'Not public' },
  'upload.warnings': { zh: '提示：{list}', en: 'Note: {list}' },

  // the photos this browser shared here, and withdrawing them
  'shared.text': {
    zh: '你在这台设备上公开了 {n} 张照片。撤回后，别人就看不到这些照片和它们的拍摄位置了。',
    en: {
      one: 'You shared {n} photo from this device. Once you withdraw it, others can no longer see it or where it was taken.',
      other: 'You shared {n} photos from this device. Once you withdraw them, others can no longer see them or where they were taken.',
    },
  },
  'shared.listLabel': { zh: '公开的照片', en: 'Your shared photos' },
  'shared.unnamed': { zh: '照片', en: 'Photo' },
  'shared.withdraw': { zh: '撤回', en: 'Withdraw' },
  'shared.withdrawAll': { zh: '全部撤回', en: 'Withdraw all' },
  // what a withdrawal did, clause by clause (common.sentence closes the last one)
  'shared.sep': { zh: '；', en: '. ' },
  'shared.withdrawn': { zh: '已撤回 {n} 张照片', en: { one: 'Withdrew {n} photo', other: 'Withdrew {n} photos' } },
  'shared.lost': {
    zh: '{n} 张服务器已找不到，无法撤回',
    en: {
      one: '{n} photo could not be withdrawn: the server no longer has it',
      other: '{n} photos could not be withdrawn: the server no longer has them',
    },
  },
  'shared.refused': {
    zh: '{n} 张的撤回凭证已失效，照片和位置可能仍然公开',
    en: {
      one: 'The withdrawal token of {n} photo is no longer valid: the photo and its position may still be public',
      other: 'The withdrawal tokens of {n} photos are no longer valid: the photos and their positions may still be public',
    },
  },
  'shared.failed': {
    zh: '{n} 张撤回失败，请稍后再试',
    en: { one: '{n} photo could not be withdrawn. Please try again later', other: '{n} photos could not be withdrawn. Please try again later' },
  },

  // web/poses.js and the landmark page's card of public photo poses (D30, D32)
  'poses.title': { zh: '公开的拍摄位置', en: 'Shared photo positions' },
  'poses.col.source': { zh: '来源', en: 'Source' },
  'poses.col.position': { zh: '位置（离中心）', en: 'Position (from the centre)' },
  'poses.col.heading': { zh: '朝向', en: 'Heading' },
  'poses.col.range': { zh: '±范围', en: '± range' },
  'poses.hint': {
    zh: '上传的照片都会公开（撤回的除外）：位置取整到 1 米，朝向取整到 1°，视角取整到 5°，不显示上传者和时间；无法可靠确定位置的不显示。每次建模结束后更新。',
    en: 'Every uploaded photo is public unless withdrawn: positions are rounded to 1 m, headings to 1° and fields of view to 5°, without the uploader or the time; photos whose position cannot be determined reliably are not shown. Updated after each build.',
  },
  'poses.asOf': { zh: '{time}更新', en: 'Updated {time}' },
  'poses.noBuildYet': {
    zh: '这里还没有建模完成过；公开的照片会在第一次建模结束后显示。',
    en: 'No build has finished here yet. Public photos appear once the first build is done.',
  },
  'poses.mine': { zh: '你的', en: 'Yours' },
  // the sources of a pose (crowd3d/photoposes.py); the EXIF label is the owner's wording (D30 question 1). They fill the
  // first column of the poses table, which must leave room for the data columns on a 390 px phone: the English EXIF
  // label stays short and its hint says what was not checked.
  'poses.source.sfm': { zh: '模型重建', en: '3D reconstruction' },
  'poses.source.loc': { zh: '与模型比对', en: 'Matched to the model' },
  'poses.source.exif': { zh: '手机 GPS + 罗盘（未经模型核对）', en: 'Phone GPS + compass (unverified)' },
  'poses.sourceHint.sfm': {
    zh: '照片参与了三维重建，位置和朝向由重建算出',
    en: 'The photo was used to build the 3D model, which worked out its position and heading',
  },
  'poses.sourceHint.loc': { zh: '照片与已建好的模型比对算出', en: 'Worked out by matching the photo to the finished model' },
  'poses.sourceHint.exif': {
    zh: '照片自带的定位和指南针，没有和模型核对，范围大得多',
    en: 'The location and compass heading the phone saved with the photo, not checked against the model; the range is much wider',
  },
  'poses.sourceLine': { zh: '{source}：{hint}。', en: '{source}: {hint}.' },
  'poses.snappedNote': {
    zh: '这个模型的素模按地图轮廓对齐过：公开的拍摄位置跟着对齐了，点云没有，画在点云上会偏开，所以这里不画。请在它的风格化模型里看。',
    en: "This model's clay version was aligned to the map footprint: the shared photo positions moved with it but the point cloud did not, so on the point cloud they would be off and are not drawn here. See them in its stylized model.",
  },
  // a position from the landmark centre, by the quarter it lies in
  'poses.position.en': { zh: '东 {ew} 米，北 {ns} 米', en: '{ew} m east, {ns} m north' },
  'poses.position.es': { zh: '东 {ew} 米，南 {ns} 米', en: '{ew} m east, {ns} m south' },
  'poses.position.wn': { zh: '西 {ew} 米，北 {ns} 米', en: '{ew} m west, {ns} m north' },
  'poses.position.ws': { zh: '西 {ew} 米，南 {ns} 米', en: '{ew} m west, {ns} m south' },
  // a heading and the direction it points to (compass.short.*)
  'poses.heading': { zh: '{deg}°（{dir}）', en: '{deg}° ({dir})' },
  'poses.range.position': { zh: '±{n} 米', en: '±{n} m' },
  'poses.counts': {
    zh: '公开的 {shared} 张 / 共 {n} 张',
    en: { one: 'Public: {shared} of {n} photo', other: 'Public: {shared} of {n} photos' },
  },
  'poses.counts.shown': {
    zh: '公开的 {shared} 张 / 共 {n} 张，显示 {shown} 张',
    en: { one: 'Public: {shared} of {n} photo, {shown} shown', other: 'Public: {shared} of {n} photos, {shown} shown' },
  },
  'poses.counts.undetermined': {
    zh: '公开的 {shared} 张 / 共 {n} 张；{undetermined} 张无法可靠确定位置和朝向，不显示',
    en: {
      one: 'Public: {shared} of {n} photo; {undetermined} not shown (position and heading could not be determined reliably)',
      other: 'Public: {shared} of {n} photos; {undetermined} not shown (position and heading could not be determined reliably)',
    },
  },
  'poses.counts.shownUndetermined': {
    zh: '公开的 {shared} 张 / 共 {n} 张，显示 {shown} 张；{undetermined} 张无法可靠确定位置和朝向，不显示',
    en: {
      one: 'Public: {shared} of {n} photo, {shown} shown; {undetermined} not shown (position and heading could not be determined reliably)',
      other: 'Public: {shared} of {n} photos, {shown} shown; {undetermined} not shown (position and heading could not be determined reliably)',
    },
  },
  // how often the ± ranges held on the benchmark's test landmarks (uncertainty.measured_coverage), per source
  'poses.coverage': {
    zh: '±范围是 {sigmas}σ。在基准的测试地标上实测，真实位置和朝向都落在范围内的比例如下。{sources}',
    en: '± ranges are {sigmas}σ. On the benchmark test landmarks, this is how often the true position and heading both fell within the range. {sources}',
  },
  'poses.coverage.source': { zh: '{source}：{coverage}。', en: '{source}: {coverage}.' },
  'poses.share': { zh: '{n} 个中 {within} 个（{percent}）', en: '{within} of {n} ({percent})' },
  'poses.measured.none': { zh: '还没有实测', en: 'not measured yet' },
  'poses.measured': { zh: '测试地标 {share}', en: '{share} on the test landmarks' },
  'poses.measured.landmarks': {
    zh: '{n} 个测试地标 {share}',
    en: { one: '{share} on {n} test landmark', other: '{share} on {n} test landmarks' },
  },
  'poses.measured.runs': { zh: '测试地标、{runs} 次运行合计 {share}', en: '{share} on the test landmarks over {runs} runs' },
  'poses.measured.landmarksRuns': {
    zh: '{n} 个测试地标、{runs} 次运行合计 {share}',
    en: { one: '{share} on {n} test landmark over {runs} runs', other: '{share} on {n} test landmarks over {runs} runs' },
  },
  'poses.measured.spread': {
    zh: '{text}，单个地标单次运行最低 {lowest}、最高 {highest}',
    en: '{text}; for a single landmark and run, lowest {lowest}, highest {highest}',
  },
  'poses.measured.calibratedHere': {
    zh: '{text}；这个来源的范围是用包括这些地标的照片定的，所以这个数不是独立检验，实际可能更低',
    en: "{text}. This source's range was set using photos that include these landmarks, so this is not an independent check and the real share may be lower",
  },
  'poses.measured.notCalibratedHere': {
    zh: '{text}；这个来源的范围规则没有用这些地标调过',
    en: "{text}. This source's range rule was not tuned on these landmarks",
  },

  // web/stylized.html and web/stylized.js: the clay viewer (D29: a clay model, never coloured)
  'stylized.docTitle': { zh: '卡通地标 · Crowd3D 众拍建模', en: 'Cartoon landmark · Crowd3D' },
  'stylized.metaDescription': {
    zh: '按开放地图数据搭出、用打卡照片校正尺寸的卡通素模地标，不上色。',
    en: 'A cartoon clay model of a landmark, built from open map data, with its sizes corrected by check-in photos. Not coloured.',
  },
  'stylized.stageLabel': {
    zh: '卡通风格三维地标模型，可拖动旋转、双指缩放，点按墙面查看来源',
    en: 'Cartoon-style 3D model of the landmark: drag to rotate, pinch to zoom, tap a wall to see where it comes from',
  },
  // the panel's title when the model names no landmark
  'stylized.landmark': { zh: '地标', en: 'Landmark' },
  'stylized.chip.walls': { zh: '立面 {n}', en: { one: '{n} wall', other: '{n} walls' } },
  'stylized.chip.roofFaces': { zh: '屋顶面 {n}', en: { one: '{n} roof face', other: '{n} roof faces' } },
  'stylized.chip.relief': { zh: '立面起伏 {n} 处', en: { one: '{n} relief feature', other: '{n} relief features' } },
  'stylized.about': {
    zh: '素模按地图轮廓和高度搭出来，照片对齐时再用照片量出的尺寸校正，不上色。点按任意一块查看来源。',
    en: "The clay model is built from the map's outlines and heights; when the photos are aligned, the sizes they measured correct it. It is not coloured. Tap any part to see where it comes from.",
  },
  'stylized.noModel': { zh: '缺少模型：在地址后加 ?model=模型.glb', en: 'No model: add ?model=model.glb to the address' },
  'stylized.error': { zh: '出错了：{message}', en: 'Something went wrong: {message}' },
  // the style picker (画风: the look, independent of the page language and of light / dark) and its notes; {error} is the
  // theme validator's technical message (web/themes.js), not translated
  'stylized.theme.label': { zh: '画风', en: 'Style' },
  'stylized.theme.invalid': {
    zh: '画风“{name}”没有通过校验，已改用默认画风：{error}',
    en: 'The style "{name}" failed validation, so the default style is used: {error}',
  },
  'stylized.theme.sep': { zh: '；', en: '; ' },
  // the card of a tapped face: its kind, then where its outline, height and shape come from (crowd3d.stylize extras)
  'stylized.kind.wall': { zh: '立面', en: 'Wall' },
  'stylized.kind.roof': { zh: '屋顶', en: 'Roof' },
  'stylized.kind.roof-cut': { zh: '屋顶切面', en: 'Roof cut' },
  'stylized.kind.corners': { zh: '转角', en: 'Corner' },
  'stylized.kind.base': { zh: '底面', en: 'Base' },
  'stylized.pick.outline': { zh: '轮廓：{source}', en: 'Outline: {source}' },
  'stylized.pick.height': { zh: '高度：{source}', en: 'Height: {source}' },
  // older models: a height carried over from another wall of the block
  'stylized.pick.heightCarried': {
    zh: '高度：{source}（取自同一体块已测的墙）',
    en: 'Height: {source} (from a measured wall of the same block)',
  },
  'stylized.pick.unknown': { zh: '未知', en: 'Unknown' },
  // provenance codes (crowd3d/stylize.py, structure.py); the codes with spaces and brackets have a key of their own
  // (stylized.js PROVENANCE_KEYS). default-style: neither measured nor tagged, and since D29 every face's appearance:
  // it never says anything about photos or colour.
  'stylized.provenance.osm-footprint': { zh: 'OSM 轮廓', en: 'OSM outline' },
  'stylized.provenance.sfm-points': { zh: '重建点云实测', en: 'Measured on the reconstructed point cloud' },
  'stylized.provenance.osm-height': { zh: 'OSM 高度标签', en: 'OSM height tag' },
  'stylized.provenance.wikidata-height': { zh: 'Wikidata 高度', en: 'Wikidata height' },
  'stylized.provenance.inferred': {
    zh: '推断（同一体块别处实测，这里没测到）',
    en: 'Inferred (measured elsewhere on the same block, not here)',
  },
  'stylized.provenance.default-style': { zh: '默认样式（没有实测，也没有标注）', en: 'Default style (neither measured nor tagged)' },
  'stylized.provenance.osm-s3db': { zh: 'OpenStreetMap 志愿者测绘的分件', en: 'Mapped by OpenStreetMap volunteers (building part)' },
  'stylized.provenance.osm-levels': { zh: 'OSM 层数换算（每层 3 米）', en: 'Converted from the OSM level count (3 m per level)' },
  'stylized.provenance.osm-roof-angle': { zh: 'OSM 屋顶坡度换算', en: 'Converted from the OSM roof angle' },
  'stylized.provenance.s3db-shape-default': {
    zh: '屋顶坡度按惯例（地图只标了屋顶形状）',
    en: 'Conventional roof pitch (the map only gives the roof shape)',
  },
  'stylized.provenance.osm-s3db-support': { zh: '由压在上面的分件推出', en: 'Inferred from the parts resting on it' },
  'stylized.provenance.osm-default-ground': { zh: '地图默认（从地面起）', en: 'Map default (from the ground)' },
  'stylized.provenance.curator': { zh: '编辑核对', en: 'Checked by an editor' },
  'stylized.provenance.photo-fitted': { zh: '照片实测（改了地图值）', en: 'Measured from photos (changed the map value)' },
  // D15: at least two photo channels agreed; the map's value is kept beside it and the change can be undone
  'stylized.provenance.photo-fitted-auto': { zh: '照片量出（自动采用）', en: 'Measured from photos (adopted automatically)' },
  // D25: strong photo evidence corrected the map (the full photo value, or only its guaranteed lower bound)
  'stylized.provenance.photo-corrected-auto': { zh: '照片纠正地图（自动）', en: 'Photos corrected the map (automatically)' },
  'stylized.provenance.photo-corrected-auto-lower-bound': {
    zh: '照片纠正地图（自动，下界）',
    en: 'Photos corrected the map (automatically, lower bound)',
  },
  // a structure part's shape (extras.geometry)
  'stylized.geometry.map': { zh: '形状：OpenStreetMap 志愿者测绘', en: 'Shape: mapped by OpenStreetMap volunteers' },
  'stylized.geometry.map-convention': {
    zh: '形状：地图事实加惯例（屋顶坡度按惯例）',
    en: 'Shape: map facts plus convention (conventional roof pitch)',
  },
  'stylized.geometry.photo-verified': { zh: '形状：已由照片核验', en: 'Shape: verified by photos' },
  'stylized.geometry.photo-fitted': { zh: '形状：照片实测', en: 'Shape: measured from photos' },
  'stylized.geometry.photo-corrected': { zh: '形状：照片纠正了地图', en: 'Shape: photos corrected the map' },
  'stylized.geometry.disputed': { zh: '形状：OpenStreetMap 志愿者测绘', en: 'Shape: mapped by OpenStreetMap volunteers' },
  'stylized.geometry.curator': { zh: '形状：编辑核对', en: 'Shape: checked by an editor' },
  'stylized.geometry.carved': { zh: '高度由照片测得，屋顶形状未知', en: 'Height measured from photos; roof shape unknown' },
  'stylized.geometry.stub': { zh: '高度待定（地图没有高度）', en: 'Height pending (the map has no height)' },
  // {label}: stylized.provenance.photo-fitted-auto
  'stylized.pick.shapeAuto': { zh: '形状：{label}', en: 'Shape: {label}' },
  'stylized.pick.framed': {
    zh: '{text}（轮廓按照片整体缩放、平移：{label}）',
    en: '{text} (outline scaled and shifted as a whole to fit the photos: {label})',
  },
  // an outline the frame adoption scaled and shifted (D15 ⑥); {k}: '×0.818'
  'stylized.pick.outlineScaled': {
    zh: '轮廓：{label}：整栋缩放 {k}（地图原轮廓宽 {mapWidth}，现 {width}，可撤销）',
    en: 'Outline: {label}: the whole building scaled {k} (the map outline was {mapWidth} wide, now {width}; can be undone)',
  },
  'stylized.pick.outlineScaledShifted': {
    zh: '轮廓：{label}：整栋缩放 {k}、平移 {shift}（地图原轮廓宽 {mapWidth}，现 {width}，可撤销）',
    en: 'Outline: {label}: the whole building scaled {k} and shifted {shift} (the map outline was {mapWidth} wide, now {width}; can be undone)',
  },
  // the part's values a photo channel changed: which one, at the start of a line
  'stylized.param.height': { zh: '顶高', en: 'Top height' },
  'stylized.param.min_height': { zh: '底高', en: 'Base height' },
  'stylized.param.roof_height': { zh: '屋顶高', en: 'Roof height' },
  'stylized.pick.adopted': { zh: '{name}：{label} {value}（地图原值 {map}，可撤销）', en: '{name}: {label}, {value} (map value {map}; can be undone)' },
  'stylized.pick.adoptedCarried': {
    zh: '{name}：随下面的分件一起移动 {value}（地图原值 {map}，可撤销）',
    en: '{name}: moved with the part below, {value} (map value {map}; can be undone)',
  },
  // D25 ({label}: stylized.provenance.photo-corrected-auto or its lower-bound form)
  'stylized.pick.correctedCarried': {
    zh: '{name}：随下面被纠正的分件一起移动 {value}（地图原值 {map}，可撤销）',
    en: '{name}: moved with the corrected part below, {value} (map value {map}; can be undone)',
  },
  'stylized.pick.correctedBlock': {
    zh: '照片量出：至少 {value}（屋顶形状未知；地图上高度待定 {map}，可撤销）',
    en: 'Measured from photos: at least {value} (roof shape unknown; height pending on the map, {map}; can be undone)',
  },
  'stylized.pick.correctedBound': {
    zh: '{label}：{name}至少 {value}（地图 {map}，可撤销）',
    en: '{label}: {name} at least {value} (map {map}; can be undone)',
  },
  'stylized.pick.correctedBoundPhoto': {
    zh: '{label}：{name}至少 {value}（地图 {map}，照片值 {photo}，可撤销）',
    en: '{label}: {name} at least {value} (map {map}, photo value {photo}; can be undone)',
  },
  'stylized.pick.corrected': {
    zh: '{label}：{name} 地图 {map} → 照片 {value}（可撤销）',
    en: '{label}: {name} {map} on the map → {value} from photos (can be undone)',
  },
  'stylized.pick.correctedRange': {
    zh: '{label}：{name} 地图 {map} → 照片 {value}（范围 {low}–{high}）（可撤销）',
    en: '{label}: {name} {map} on the map → {value} from photos (range {low}–{high}; can be undone)',
  },
  'stylized.pick.tilt': { zh: '倾斜：照片量出 {tilt}°（地图上是直的，可撤销）', en: 'Lean: {tilt}° measured from photos (upright on the map; can be undone)' },
  'stylized.pick.tiltToward': {
    zh: '倾斜：照片量出 {tilt}°，朝 {azimuth}°（地图上是直的，可撤销）',
    en: 'Lean: {tilt}° towards {azimuth}°, measured from photos (upright on the map; can be undone)',
  },
  // {tagged}, {shape}: OpenStreetMap roof:shape values, as the map has them
  'stylized.pick.roofFallback': {
    zh: '屋顶：地图标的是 {tagged}，搭不出来，画成 {shape}',
    en: 'Roof: the map tags it {tagged}, which could not be built, so it is drawn {shape}',
  },
  // facade relief the photos measured on a wall (crowd3d.relief kinds): one of a kind, and several (the same in Chinese)
  'stylized.relief.arch': { zh: '拱', en: 'arch' },
  'stylized.relief.arched-bay': { zh: '拱券开间', en: 'arched bay' },
  'stylized.relief.bay': { zh: '凹开间', en: 'recessed bay' },
  'stylized.relief.panel': { zh: '凹板', en: 'recessed panel' },
  'stylized.relief.window': { zh: '窗洞', en: 'window opening' },
  'stylized.relief.arched-window': { zh: '拱窗', en: 'arched window' },
  'stylized.relief.door': { zh: '门洞', en: 'doorway' },
  'stylized.relief.arched-door': { zh: '拱门', en: 'arched doorway' },
  'stylized.relief.setback': { zh: '退台', en: 'setback' },
  'stylized.relief.groove': { zh: '凹槽', en: 'groove' },
  'stylized.relief.band': { zh: '腰线', en: 'string course' },
  'stylized.relief.plinth': { zh: '勒脚', en: 'plinth' },
  'stylized.relief.projecting-zone': { zh: '凸出带', en: 'projecting band' },
  'stylized.relief.pilaster': { zh: '壁柱', en: 'pilaster' },
  'stylized.relief.buttress': { zh: '扶壁', en: 'buttress' },
  'stylized.relief.corner-pier': { zh: '角柱', en: 'corner pier' },
  'stylized.reliefs.arch': { zh: '拱', en: 'arches' },
  'stylized.reliefs.arched-bay': { zh: '拱券开间', en: 'arched bays' },
  'stylized.reliefs.bay': { zh: '凹开间', en: 'recessed bays' },
  'stylized.reliefs.panel': { zh: '凹板', en: 'recessed panels' },
  'stylized.reliefs.window': { zh: '窗洞', en: 'window openings' },
  'stylized.reliefs.arched-window': { zh: '拱窗', en: 'arched windows' },
  'stylized.reliefs.door': { zh: '门洞', en: 'doorways' },
  'stylized.reliefs.arched-door': { zh: '拱门', en: 'arched doorways' },
  'stylized.reliefs.setback': { zh: '退台', en: 'setbacks' },
  'stylized.reliefs.groove': { zh: '凹槽', en: 'grooves' },
  'stylized.reliefs.band': { zh: '腰线', en: 'string courses' },
  'stylized.reliefs.plinth': { zh: '勒脚', en: 'plinths' },
  'stylized.reliefs.projecting-zone': { zh: '凸出带', en: 'projecting bands' },
  'stylized.reliefs.pilaster': { zh: '壁柱', en: 'pilasters' },
  'stylized.reliefs.buttress': { zh: '扶壁', en: 'buttresses' },
  'stylized.reliefs.corner-pier': { zh: '角柱', en: 'corner piers' },
  // {kind}: stylized.relief.* (one) or stylized.reliefs.* (several); the items of a list
  'stylized.reliefCount': { zh: '{kind} {n}', en: '{n} {kind}' },
  // with a 3-D line at an edge, checked by both photo halves
  'stylized.pick.reliefLined': {
    zh: '立面起伏：照片实测 {n} 处（{kinds}），每处两半照片各自量到，并有三维直线核对',
    en: {
      one: 'Relief measured by photos: {n} feature ({kinds}), found separately in both halves of the photos and checked against a 3-D line',
      other: 'Relief measured by photos: {n} features ({kinds}), each found separately in both halves of the photos and checked against a 3-D line',
    },
  },
  // D24 (the product owner's decision of 2026-10-24): on the photos' point cloud alone; {label}: stylized.reliefD24
  'stylized.reliefD24': { zh: '照片点云量出（D24）', en: 'measured on the photo point cloud alone (D24)' },
  'stylized.pick.reliefD24': {
    zh: '立面起伏：{label} {n} 处（{kinds}），两种照片分法都量到，原始点重量对得上；这面墙的三维直线到处都是（处于偶然水平），没有直线核对',
    en: {
      one: 'Relief {label}: {n} feature ({kinds}). Both ways of splitting the photos found it and re-measuring the raw points agrees; this wall has 3-D lines everywhere (at chance level), so it was not checked against a line',
      other: 'Relief {label}: {n} features ({kinds}). Both ways of splitting the photos found them and re-measuring the raw points agrees; this wall has 3-D lines everywhere (at chance level), so they were not checked against a line',
    },
  },
  'stylized.pick.reliefUnbuilt': {
    zh: '另实测 {n} 处不单独建模（如两个凹开间之间的壁柱就是墙面本身）',
    en: {
      one: '{n} more feature measured but not modelled on its own (a pilaster between two recessed bays, for example, is the wall itself)',
      other: '{n} more features measured but not modelled on their own (a pilaster between two recessed bays, for example, is the wall itself)',
    },
  },

  // web/themes.js: the default style's names (DEFAULT_THEME = themes/default.json; the other styles carry theirs in
  // their JSON: name, en.name) and why a requested style could not be used
  'themes.default.name': { zh: '明快扁平', en: 'Bright flat' },
  'themes.default.description': {
    zh: '明亮的扁平色块配粗墨线描边，三档卡通明暗，现在的默认外观。',
    en: 'Bright flat colours with thick ink outlines and three-step cartoon shading: the default look.',
  },
  'themes.error.badId': { zh: '画风名称无效：{id}', en: 'Not a valid style name: {id}' },
  'themes.error.wrongId': { zh: '画风文件 {id}.json 的 id 不是 {id}', en: 'The style file {id}.json does not have the id {id}' },
  'themes.error.loadFailed': { zh: '画风 {id} 加载失败：{message}', en: 'The style {id} could not be loaded: {message}' },

  // web/photoposes.js: the public photo poses on the clay model (D30, D32). The sources as the layer names them: the
  // EXIF label keeps the owner's full wording in both languages (the legend shows it without its hint; the landmark
  // page's table uses the shorter poses.source.exif)
  'pp.source.sfm': { zh: '模型重建', en: '3D reconstruction' },
  'pp.source.loc': { zh: '与模型比对', en: 'Matched to the model' },
  'pp.source.exif': { zh: '手机 GPS + 罗盘（未经模型核对）', en: 'Phone GPS and compass (not checked against the model)' },
  // a row's 2-sigma ranges, and how the server rounded (doc.rounding)
  'pp.error': { zh: '位置 ±{sp} 米，朝向 ±{sy}°', en: 'position ±{sp} m, heading ±{sy}°' },
  'pp.rounding': {
    zh: '位置取整到 {m} 米、朝向取整到 {d}°、视角取整到 {f}°',
    en: 'position rounded to {m} m, heading to {d}°, field of view to {f}°',
  },
  'pp.rangeSpan': { zh: '±{lo}–{hi} 米', en: '±{lo}–{hi} m' },
  // while looking from a photo's pose; Off: its range is wider than the disc threshold
  'pp.banner': {
    zh: '从这张照片的位置和视角看（{source}）：{error}（2σ），{rounding}',
    en: "From this photo's position and field of view ({source}): {error} (2σ); {rounding}.",
  },
  'pp.bannerOff': {
    zh: '从这张照片的位置和视角看（{source}）：{error}（2σ），{rounding}；相机可能在这个范围里任何地方，看到的画面可能和照片对不上',
    en: "From this photo's position and field of view ({source}): {error} (2σ); {rounding}. The camera may be anywhere in this range, so the view may not match the photo.",
  },
  // the card of one photo
  'pp.card.title': { zh: '一张公开的照片', en: 'A public photo' },
  'pp.card.source': { zh: '来源：{source}。{hint}', en: 'Source: {source}. {hint}.' },
  'pp.card.pos': { zh: '位置：离地标中心{position}；±{sp} 米', en: 'Position: {position} of the landmark centre; ±{sp} m' },
  'pp.card.posHeight': {
    zh: '位置：离地标中心{position}，高 {height} 米；±{sp} 米',
    en: 'Position: {position} of the landmark centre, at a height of {height} m; ±{sp} m',
  },
  'pp.card.posNoHeight': {
    zh: '位置：离地标中心{position}（拍摄高度没有测出）；±{sp} 米',
    en: 'Position: {position} of the landmark centre (the camera height was not measured); ±{sp} m',
  },
  'pp.card.dir': { zh: '朝向：{heading}，±{sy}°', en: 'Heading: {heading}, ±{sy}°' },
  'pp.card.dirNoPitch': { zh: '朝向：{heading}，±{sy}°，仰角没有测出', en: 'Heading: {heading}, ±{sy}°; the tilt was not measured' },
  'pp.card.dirUp': { zh: '朝向：{heading}，±{sy}°，仰角 {pitch}°', en: 'Heading: {heading}, ±{sy}°, tilted {pitch}° up' },
  'pp.card.dirDown': { zh: '朝向：{heading}，±{sy}°，俯角 {pitch}°', en: 'Heading: {heading}, ±{sy}°, tilted {pitch}° down' },
  'pp.card.fov': { zh: '视角：水平约 {hfov}°', en: 'Field of view: about {hfov}° across' },
  'pp.card.fovNone': { zh: '视角：没有测出', en: 'Field of view: not measured' },
  // {coverage}: poses.js sourceCoverageText
  'pp.card.range': {
    zh: '±范围是 {sigmas}σ。这个来源在基准测试地标上实测，真实位置和朝向都落在范围内的：{coverage}。',
    en: '± ranges are {sigmas}σ. How often the true position and heading both fell within the range for this source, measured on the benchmark test landmarks: {coverage}.',
  },
  // the card's and the legend's last line ({rounding}: pp.rounding)
  'pp.privacy': {
    zh: '{rounding}；不显示是谁拍的、什么时候拍的；照片本身去掉拍摄信息（EXIF）并缩小后公开。',
    en: 'Shown with the {rounding}. Who took the photo and when are not shown; the photo itself is public, with its capture data (EXIF) removed and scaled down.',
  },
  'pp.viewFrom': { zh: '从这张照片的角度看', en: 'View from this photo' },
  // why "view from this photo" is off (D30: never an assumed height, tilt or field of view)
  'pp.blocked.exif': {
    zh: '手机 GPS + 罗盘只给出大致的位置和朝向，没有测出拍摄高度和仰角，无法还原拍摄角度',
    en: "the phone's GPS and compass give only a rough position and heading; the camera's height and tilt weren't measured",
  },
  'pp.blocked.pose': { zh: '没有测出拍摄高度、仰角或侧倾，无法还原拍摄角度', en: "the camera's height, tilt or roll wasn't measured" },
  'pp.blocked.fov': {
    zh: '没有测出视角（焦距或画幅未知），无法还原拍摄角度',
    en: "the field of view wasn't measured (unknown focal length or frame size)",
  },
  'pp.blockedLine': { zh: '不能从这张照片的角度看：{reason}。', en: "Can't view from this photo: {reason}." },
  'pp.close': { zh: '关闭', en: 'Close' },
  'pp.prev': { zh: '上一张（绕地标逆时针）', en: 'Previous photo (anticlockwise around the landmark)' },
  'pp.next': { zh: '下一张（绕地标顺时针）', en: 'Next photo (clockwise around the landmark)' },
  'pp.back': { zh: '返回总览', en: 'Back to the overview' },
  // the legend
  'pp.legend.title': { zh: '照片的拍摄位置和朝向', en: 'Photo positions and headings' },
  'pp.legend.label': { zh: '照片拍摄位置图例', en: 'Legend of the photo positions' },
  // a source's count, and how many of them this page could draw when that differs
  'pp.legend.countDrawn': { zh: '{n} · 画出 {drawn}', en: '{n} · {drawn} drawn' },
  'pp.legend.shown': { zh: '显示 {n} 张', en: { one: 'Showing {n} photo', other: 'Showing {n} photos' } },
  'pp.legend.allRanges': { zh: '显示所有照片的范围', en: "Show every photo's range" },
  'pp.legend.show': { zh: '在模型上显示照片', en: 'Show the photos on the model' },
  'pp.legend.disc': {
    zh: '范围超过 {n} 米的照片在地上画出范围圆（2σ）：相机可能在圆里任何地方，不一定在画出的那一点。',
    en: 'Photos with a range over {n} m have their range circle (2σ) drawn on the ground: the camera may be anywhere in the circle, not necessarily at the point drawn.',
  },
  'pp.legend.asOf': { zh: '{time} 更新（这里每次建模结束后更新）', en: 'Updated {time} (after each build here)' },
  'pp.legend.dropped': {
    zh: '{n} 行数据不完整，没有画',
    en: { one: '{n} photo not drawn: its data is incomplete', other: '{n} photos not drawn: their data is incomplete' },
  },
  // the north label at the edge of the plaza (Chinese also gives the letter)
  'pp.north': { zh: '北 N', en: 'N' },
  // a cluster bubble's tooltip: the count per source, then what a tap does
  'pp.bubble.count': { zh: '{source} {n}', en: '{source}: {n}' },
  'pp.bubble.title': { zh: '{counts}（点按放大）', en: '{counts} (tap to zoom in)' },
  // the hover tip of a photo
  'pp.tip': { zh: '{source} · 朝向 {heading} · {error}', en: '{source} · heading {heading} · {error}' },
};
