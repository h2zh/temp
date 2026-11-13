# HTTP API

> 原来是 README 的"HTTP API"一节，D34 C 部分第 2 步（2026-11-13）搬到这里，文字没有改，只在路由表里补了 `/openapi.json` 一行。
> 下面的路由表由 `tests/test_docs.py`（快速测试）和应用真正注册的路由对：服务提供的每条路由都要在表里，表里也不能有服务没有的路由。比的是方法和路径（路径参数的名字不比：表里写 `{id}`，代码里是 `{landmark_id}`）；"说明"一列是手写的。

除了照片接口（JPEG），所有接口都是 JSON。时间是 unix 秒，角度是度，距离是米。字段的完整定义见 `crowd3d/api.py`。

API 只发代码和英文的技术说明，没有中文（D32）：拒绝码、警告码、作业状态、作业错误和 422 的 `type` 都由网页按页面语言翻译（`web/api.js`）。用户自己写的内容（地标名、昵称、文件名）原样返回。

所有接口共用的规则（D34）：
- 请求体上限按路由分：只有 `POST /api/photos` 可以到 50 × 25 MB 加 1 MiB，其余请求一律 64 KiB。超出返回 413 `{"detail": "request body over N bytes"}`：带 `Content-Length` 的不读请求体就拒绝，不带的读到超过上限为止，都不解析。不带 `Content-Length` 的请求体如果在服务器已经开始回答之后才超过上限，就不是 413 了（实际只有带着分块请求体去取文件的 GET：发文件时服务器会读请求体）。这时回答照常发完，超出的请求体读了就丢，不交给应用；这样的请求得到 413 还是整个文件，看请求体和回答谁先到（D34 整体验证之后）。发压缩形式的文件时（见下面"压缩"一条）先读完请求体再回答，所以超限一定是 413。
- 上传接口的表单另有自己的上限（D34）：文本字段最多 8 个、每个最多 4 KiB，文件部分最多 50 个；一个文件部分超过 64 KiB 就写到临时文件，不放在内存里。解析到第一个超限的地方就停，后面的请求体不读。文本字段超限返回 413 `{"detail": "upload form over 8 text fields, or a text field over 4096 bytes"}`；文件超过 50 个时的回答和顺序同以前（先看凭证，再说文件个数），只是凭证要排在第 51 个文件之前才读得到：客户端要把 `contributor_token` 放在文件前面（网页是这么发的），不然登记过的人传 51 个文件得到的是 `register_required`。不是 multipart 的请求体不读，照旧是 422（缺 `files`）。
- 422 的每条错误最多回显 1 KiB 的 `input`（按响应里的 JSON 字节算），超过就是 null。请求体里可能有登记凭证的接口一律不回显：上传、撤回、登记名下的照片、同意新版本的 422 `input` 都是 null（登记接口只回显文字的昵称）。404 里引用的地标 id、作业 id 最多 80 个字符。
- 每个响应都带 `X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`、`Referrer-Policy: strict-origin-when-cross-origin` 和 `Content-Security-Policy`。页面的策略只允许本站的脚本、样式和请求，内联块按哈希放行，图片另允许 `data:`、`blob:` 和地图瓦片；其余响应（JSON、照片、模型文件、错误）一律 `default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`。
- FastAPI 自带的 `/docs` 和 `/redoc` 不提供（它们从 CDN 加载脚本）；`/openapi.json` 照旧。
- 网页文件（`web/` 下的每个文件，含 `web/vendor/` 里的库）带 `Cache-Control: no-cache`：浏览器每次用之前凭 ETag 问一次，没变返回 304。模型文件和照片接口的缓存规则各管各的。
- 压缩（D34 第 4 步）：网页的文本文件（`.html`、`.js`、`.mjs`、`.css`、`.json`、`.svg`，含 `web/vendor/` 里的库）和 `/files/models` 下的 `.glb`、`.ply`，在请求的 `Accept-Encoding` 允许 gzip、文件在 1 KiB 到 8 MiB 之间、请求不带 `Range` 时压缩着发（`Content-Encoding: gzip`，带准确的 `Content-Length`）。一个文件的两种形式各有 ETag（压缩的那个在引号里多 `-gzip`），都带 `Vary: Accept-Encoding`；`If-None-Match` 和这个请求会得到的那种形式的 ETag 比。带 `Range` 的请求得到没压缩的字节。API 的 JSON、错误响应、照片接口和别的文件类型（PNG 等）不压缩。只有 gzip，没有 brotli。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | `{"ok": true, "reconstructor": "colmap"}` |
| GET | `/api/landmarks` | `{"landmarks": [LandmarkSummary]}`：进度、扇区状态、最新全景模型 |
| POST | `/api/landmarks` | `{id?, name, lat, lon, radius_m?}` → 201 LandmarkSummary；id 重复返回 409，参数非法返回 422。不给 id 时由名称生成 slug，中文名生成 `lm-xxxxxxxx` |
| GET | `/api/landmarks/{id}` | LandmarkDetail：在 Summary 的基础上加共同完成者（只有昵称，按昵称排序，不给每人的照片数，D30）、每个类型/扇区的最新模型、最近 20 个重建作业（风格化作业不在列表里）、阈值；不存在返回 404。不含逐张照片的信息（D30：谁在哪儿拍的、相机的精确位姿都不公开） |
| GET | `/api/landmarks/{id}/poses` | 公开的拍摄位置（D30，`crowd3d.photo-poses` 第 2 版，见下面）；不存在返回 404 |
| GET | `/api/landmarks/{id}/poses/{public_id}/photo` | 一行公开拍摄位置的照片（D32）：`image/jpeg`，由处理图重新编码，去掉全部元数据（只留 JFIF 头），长边 ≤ 1600 px；`Cache-Control: no-cache`、弱 ETag `W/"<public_id>.1"`，`If-None-Match` 对上时返回 304（也要先过同样的检查），不给 `Last-Modified`。只给此刻 `/poses` 里的行、而且照片仍公开着；其余一律返回同一个 404 `{"detail": "no such photo"}`（撤回的、没通过的、快照之后上传的、没有位姿的、库内 id、别的地标的行、任何别的字符串）；地标不存在时返回地标的 404 |
| POST | `/api/contributors` | `{nickname, terms}` → 201 `{nickname, terms, contributor_token}`，`Cache-Control: no-store`：登记一位上传者（D33），用户协议在这里同意一次，不再每次上传都同意。先查 `terms`：缺了或空着返回 422 `terms_required`，别的值返回 422 `terms_outdated`（`loc` 为 `["body", "terms"]`，`ctx` 是 `{version, url: "/terms.html"}`）；再查昵称：去掉首尾空白后 1–64 个字符、不含控制字符和落单的 UTF-16 代理项，任何文字都行，可以重名，不合格返回 422（`loc` 为 `["body", "nickname"]`）。不合格时什么都不存。请求体由服务器自己读：只认 `Content-Type: application/json`、16 KiB 以内、顶层是对象的 JSON，别的形状都当作空的请求体（返回 `terms_required`）；协议版本被拒时 `input` 为 null，只回显文字的昵称（能编码成 UTF-8 的字符串）；客户端没发完请求体就断开时返回 400。`contributor_token` 是 64 位十六进制的密钥，只在这次响应里给一次，库里只存它的 SHA-256；响应里没有上传者的 id。没有 GET：任何接口都不给出登记记录，共同完成者仍只列通过的照片的昵称 |
| POST | `/api/contributors/accept` | `{contributor_token, terms}` → `{nickname, terms}`：已登记的上传者同意现在版本的协议（D33：协议出了新版本时同意一次，不用重新起昵称）。凭证缺了、不是字符串、格式不对或不认识返回 422 `register_required`；`terms` 不是现在的版本返回 422 `terms_required` / `terms_outdated`，什么都不改。请求体的规则和登记一样，别的形状（不是 JSON、表单、数组、别的 Content-Type、超过 16 KiB）都当作空的请求体（返回 `register_required`）；422 的 `input` 一律为 null，从不回显请求体里的任何内容（D33，99b57e5）；客户端没发完请求体就断开时返回 400 |
| POST | `/api/photos` | multipart：`files`（1–50 个）、`contributor_token`（登记时拿到的凭证，D33）、`landmark_id`（可选，不给时按 GPS 自动匹配地标）→ `{"results": [UploadResult], "triggered_jobs": [JobInfo]}`。凭证最先检查：缺了、空着、格式不对或不认识返回 422 `register_required`；上传者同意的协议版本不是现在的版本返回 422 `terms_outdated`（`ctx` 是 `{version, url: "/terms.html"}`，先用 `POST /api/contributors/accept` 同意）。两种都是 `loc` 为 `["body", "contributor_token"]`、`input` 为 null（响应里从不回显凭证），这时什么都不存、不排作业。这个接口的 422 一律 `input` 为 null，不管表单怎么写（凭证、照片或地标作为别的类型的部分发来、表单编码、JSON），文件数不对时也一样。照片的昵称（`user_id`）和同意（`location_consent`）都取自登记记录；D32 的表单字段 `user_id`、`terms` 不再读。UploadResult 的 `share_location` 说这张照片是否公开（D32 起每张通过的照片都是 true）；公开时 `public_id` 是这张照片在公开位置里的 id，`manage_token` 是撤回用的凭证；这次响应没送到时，上传用的登记可以从 `POST /api/contributors/photos` 再拿到这两项（D34）。照片本身的校验失败放在每个文件的结果里，不作为 HTTP 错误返回。请求本身出错时仍返回 HTTP 错误：文件数不在 1–50 返回 422，`landmark_id` 不存在返回 404，请求体超过上限、表单的文本字段超过 8 个或某个超过 4 KiB 返回 413。两种凭证（登记的和撤回的）都只放在 POST 的请求体里，不放进网址或请求头 |
| POST | `/api/contributors/photos` | `{contributor_token, landmark_id?}` → `{"photos": [{photo_id, landmark_id, public_id, manage_token}]}`，`Cache-Control: no-store`（D34）：这份登记名下现在公开着的照片（用它上传、通过、没撤回的），按上传顺序；给了 `landmark_id` 就只列那个地标的（没有这个地标时是空列表）。`manage_token` 就是上传时给过的撤回凭证，所以上传的响应丢了、浏览器把凭证弄丢了，只要登记还在就能拿回来。别人登记的照片、没通过的、撤回了的都不列。不检查登记同意的协议版本（查和撤回不以同意新协议为条件）。凭证缺了、格式不对或不认识返回 422 `register_required`；`landmark_id` 不是文字返回 422；两种的 `input` 都是 null。请求体的规则和登记一样由服务器自己读，从不回显请求体里的任何内容；客户端没发完请求体就断开时返回 400。不分页。没有 GET |
| POST | `/api/photos/{photo_id}/share_location` | `{share: false, token}` 或 `{share: false, contributor_token}` → `{photo_id, share_location: false}`：撤回这张照片（D30；D32 起同时撤下公开的位置和照片本身，马上生效）。`token` 是这张照片的 `manage_token`；`contributor_token` 是上传它的那份登记的凭证（D34，不检查登记同意的协议版本）。两个只能给一个，都给、都不给、格式不对或 `share` 不是 false 返回 422（同意只在登记时给出，D33；D32 时在每次上传时）；这个接口的 422 一律 `input` 为 null。撤不了一律返回同一个 403 `{"detail": "wrong token for this photo"}`：`manage_token` 不对（照片存不存在都一样）；用登记撤回时，照片不存在、是别人登记的、是登记功能之前存的、凭证不认识或格式不对都一样，所以回答不透露照片存不存在、是谁的。自己的照片再撤一次、或撤自己没通过的照片是 200。`manage_token` 对而照片已不在库里返回 404 |
| POST | `/api/landmarks/{id}/rebuild` | `{kind: "face"\|"full", sector?}` → 202 JobInfo；照片数没达到最低要求返回 409，kind 或 sector 非法返回 422 |
| GET | `/api/jobs/{job_id}` | JobInfo（kind 取值：face / full / stylize；status 取值：queued / running / done / failed；风格化作业的 `model_id` 是它处理的模型；失败作业的 `error` 里的服务器路径和照片的库内 id 换成 `<path>`、`<id>`，原文只留在库里） |
| GET | `/openapi.json` | FastAPI 生成的接口描述（OpenAPI 的 JSON）。交互页面 `/docs` 和 `/redoc` 不提供，见上面"所有接口共用的规则" |
| GET | `/files/models/<地标>/<模型>.ply` | 点云文件。风格化作业生成的 `.glb` 也在这里。旁边的 `.json`、`.style.json` 不提供（里面有每张照片的位姿）。两种文件对接受 gzip 的客户端都压缩着发（D34） |
| GET | `/`、`/landmark.html?id=…`、`/terms.html` | 网页（`web/` 目录，不含 `web/tests/`：那里的文件返回 404，D34）；`terms.html` 是用户协议（D32）；页面用的 three.js 和 Leaflet 在 `/vendor/` 下，由本站提供（D34） |
| GET | `/stylized.html?model=<GLB 地址>&theme=<画风>&poses=<地址>` | 风格化查看器（静态页面）。`model` 和 `poses` 只接受同源地址或相对路径，跨域地址会被忽略；`poses` 是地标的公开拍摄位置，画在素模上 |

公开的拍摄位置（`GET /api/landmarks/{id}/poses`，D30）只有一种受众，没有参数：
- 只列通过校验、没撤回的照片（`records.location_shared`：照片的同意是 `records.TERMS_VERSIONS` 里的协议版本）。D32 起上传必须同意用户协议（D33 起在登记时同意一次），所以每张通过的照片都算同意，其中确定了位置和朝向的才成为一行（见下一条）；D32 之前的照片在第一次打开数据库时迁移成同意。
- 每行只有 `id, src, p, yaw, pitch, roll, hfov, aspect, sp, sy`：`id` 是用部署密钥算的 12 位 HMAC（不是库内 id），`src` 是 `sfm`（模型重建）、`loc`（与模型比对：视觉定位作业把照片定位到地标的已存地图里，`Settings.localize_photos` 默认关，关着时没有，以前存下的定位结果也不显示）或 `exif`（只用指南针标为真北的手机 GPS + 罗盘，没有高度）；位置取整到 1 米，朝向、俯仰、横滚取整到 1°，水平视角取整到 1°（D33；之前 5°），`sp`/`sy` 是 2σ 范围，按取整放宽后向上取整。按 `id` 排序，不带上传者、时间、模型或扇区。拿不准的（只有 GPS、磁北或没标的指南针、框架没定住的模型、被标记的相机）不列，只计入 `counts.not_determined`。
- 坐标是地标 ENU（米），也就是风格化 GLB 的坐标系：风格化作业接受了外轮廓吸附的模型，相机（以及定位到它里面的照片）跟着吸附移动。
- `loc` 行来自视觉定位作业存下的结果（定位作业是内部的：不在上传结果、详情里，`/api/jobs/{id}` 对它返回 404）；结果一存下就算，但照片和其他行一样要在快照里。
- 这是一张快照：只含最近一次重建作业（立面或全景，成功或失败）结束前上传的照片（`as_of` 是那个时刻），所以单独一次上传不会让列表变化；撤回同意则马上生效。但任何人都能让下一次作业很快结束（手动重建，或者自己上传几张照片），所以一个新昵称刚出现时，紧接着出现的行仍可能和它对上；用户协议第 7 节和上传说明写明了这一点（`terms-v1`）。照片接口用的是同一张快照：一行在 `/poses` 里，它的照片才能取到。
- `uncertainty.measured_coverage` 是各来源在基准测试地标上实测落在范围内的比例（commons-ten-v1 测试集，D31 全新运行按 D30 规则重算，两次运行合计：模型重建 146/165，单栋单次最低 1/10、最高 13/13；手机 GPS + 罗盘 23/28，最低 1/2、最高 1/1；与模型比对在测试集上还没有有参考的定位位姿，没实测，而且视觉定位关着）。D30 的验证在 716de16 上重算了一遍，逐张相同。`rule_calibrated_here` 说这个来源的误差规则是不是用这些地标的照片标定的：EXIF 是（样本内，地标页写明"不是独立检验"），模型重建不是。`counts` 给出快照里通过的照片数、公开的（D32 起就是没撤回的）、显示的、各来源的和没显示的。

ModelInfo（Summary 的最新全景模型、Detail 的模型列表）有两块和风格化、地理位置有关：
- `stylized`：
  - `status` 为 `ready` 时，`url` 指向 `/files/models/…/<模型>.glb`，`registration` 是 `{accepted, reason}`（`reason` 也照上面换掉路径和库内 id；网页按 `accepted` 写"素模（照片已对齐）"或"素模（照片未对齐）"，D32 起 API 不再发中文的 `label`），`poses_url` 是地标的公开拍摄位置（风格化查看器的链接带上 `&poses=`），`snapped` 说这个 GLB 的外轮廓吸附有没有被接受（接受了，这个模型的相机在公开拍摄位置里跟着吸附移动，点云却没动，所以地标页不在它的点云上画拍摄位置，改为提示去风格化模型里看）；生成中时 `snapped` 为 `null`；
  - `status` 为 `pending` 时，风格化作业正在排队或运行；
  - 没有、也不会有风格化模型时为 null。
- `geo`：按现在的下限算出的地理锚点。
  - 字段：`level`（full / prior_only / none）、各分量的 1σ `sigma`、来源 `source`、`calibration_id`、点中位处的水平和竖直 σ `position_sigma_m`、`calibrated`；
  - `calibrated` 只在标定 id 列在 `Settings.calibrated_anchor_ids` 里时为 true。这个列表现在是空的，所以地标页不显示"位置 ±X 米（1σ）"；
  - 锚点出现之前的模型为 null。只供显示，质量门不读它。

结构框架本身没有 API，也没有"重新风格化"的接口。
