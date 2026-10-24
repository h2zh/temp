# Crowd3D：众包打卡照片 → 地标三维模型（原型）

不同的用户在著名建筑前打卡，从不同角度上传照片。系统自动判断每张照片是从建筑的哪一侧、朝哪个方向拍的。照片还不够时，先重建能拍到的那一面（**立面模型**）。照片覆盖了大部分方向后，再重建整栋建筑（**全景模型**）。模型以地标中心为原点，单位是米，北向对齐。

这是一个只用 CPU 的最小原型：后端是 Python（FastAPI + SQLite + pycolmap），前端是不需要构建的原生 JS 页面，并带一个可以生成"真值"的合成数据生成器，用于测试和演示。完整的技术架构（从原型到生产的路线、算法细节、成本、合规）见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，其中第 11 节逐个模块对照了本仓库的原型和生产架构。

## 快速开始

需要 Python ≥ 3.11（开发时用 3.13）。运行 Web 测试还需要 Node ≥ 18，可选。

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q -m "not slow"   # 快速测试，约 20 秒
python -m pytest -q                 # 加上真实 COLMAP 的慢测试（含端到端验收），约 2.5 分钟
```

启动服务，然后运行演示：

```bash
python -m crowd3d.api --seed-demo   # http://127.0.0.1:8000 ，创建地标 demo-building
python scripts/demo.py              # 另开一个终端
```

`demo.py` 会生成合成照片，然后按一个故事分阶段上传。整个过程在 4 核 CPU 上约 1 分钟：

1. 游客1 在北侧拍了 4 张 → 生成北立面模型；
2. 更多游客拍了东北、东、东南、南 → 每面各生成一个立面模型；
3. 一位游客补齐了剩下几面 → 生成剩下的立面模型和全景模型（32 张照片）；
4. 几张不合格的照片 → 逐张显示拒绝原因。

终端里会打印文字版罗盘、作业状态和模型质量。最后给出查看地址，例如 `http://127.0.0.1:8000/landmark.html?id=demo-building`。在浏览器里可以看到罗盘、共同完成者、作业列表和 three.js 点云查看器。首页 `http://127.0.0.1:8000/` 是地图和地标列表。

常用参数：

- `python -m crowd3d.api`：`--data-dir DIR`（默认 `./data` 或 `$CROWD3D_DATA_DIR`）、`--reconstructor fake`（不跑 COLMAP，秒出假模型，适合调界面）、`--host`、`--port`。
- `python scripts/demo.py`：`--users N`（参与的游客数）、`--size 800x600`（照片越大越慢，默认 640x480）、`--seed 1`（换一组照片；同一组照片重复上传会被判为重复）、`--landmark ID`（新建或使用另一个地标）、`--server URL`、`--photos DIR`。
- 自己生成照片，再从网页手动上传：`python -m crowd3d.synthetic --out /tmp/photos --size 640x480`。照片的 EXIF 带 GPS 和指南针朝向，目录里有 `manifest.json` 记录真值。

> 网页从 CDN 加载 Leaflet、three.js 和 OSM 地图瓦片，所以浏览器需要能访问外网。

## 截图

以下截图来自真实服务（COLMAP 重建）：先运行 `scripts/demo.py`，再在网页上传 2 张合格照片和 3 张不合格照片。截图环境不能访问外网，所以地图底图是灰色占位。

**地标页（桌面）**：
- 左侧是 8 个方向的覆盖罗盘：数字为照片数，✓ 表示已生成立面模型。下面是上传结果，未通过的照片逐张显示原因。
- 右侧是全景模型的点云，以地标中心为原点、北向对齐。红色 N 箭头和右上角的指南针指向北，橙色点是重建估计出的拍摄位置。

![地标页（桌面）](docs/screenshots/landmark-desktop.png)

**手机版地标页（单列布局）** 和 **首页（地图、地标卡片、新建地标）**：

<p>
  <img src="docs/screenshots/landmark-mobile.png" alt="手机版地标页" width="300">
  <img src="docs/screenshots/index-desktop.png" alt="首页" width="520">
</p>

## 工作原理

```
上传 ─▶ 读 EXIF ─▶ 校验 ─▶ 入库（扇区）─▶ 覆盖规划 ─▶ 重建作业（COLMAP）─▶ 地理配准 ─▶ PLY + 元数据 ─▶ 网页
```

1. **读 EXIF**（`exif.py`）：GPS、指南针朝向（`GPSImgDirection`）、焦距、机型、拍摄时间。
2. **校验**（`validation.py`）：所有不通过的原因一起返回，不会遇到第一个就停。拒绝码有：
   - `no_gps`、`no_landmark`、`too_far`（超出地标半径）
   - `facing_away`（指南针朝向偏离"相机→地标"方向超过 60°）
   - `too_small`（短边 < 480 px）
   - `too_blurry`（拉普拉斯方差，按 800 px 归一化）
   - `duplicate`（dHash 汉明距离 ≤ 4）
   - `unreadable_image`
   - 另外，`ingest.py` 在文件超过大小上限时给出 `too_large`。
3. **扇区**（`coverage.py`）：地标四周分成 8 个罗盘扇区（N、NE……NW）。照片归入"地标中心→拍摄者"方位所在的扇区，也就是拍摄者看到的那一面。
4. **覆盖规划**（`coverage.plan_jobs`，纯函数）：
   - 某个扇区有 ≥ 4 张合格照片时，触发**立面作业**，输入是该扇区的照片，加上相邻扇区 10° 以内的照片。
   - ≥ 6 个扇区就绪且总数 ≥ 24 张时，触发**全景作业**。
   - 之后照片继续增长，立面每多 3 张、全景每多 8 张就重建一次。失败的作业也计入，要等再增长够了才会重试。
5. **作业**（`jobs.py`）：单个后台线程按顺序执行（SfM 很吃 CPU）。作业做完后会重新规划一次，所以作业运行期间新到的照片不会被漏掉。服务重启时，未完成的作业会重新排队。
6. **重建**（`reconstruct.py`）：pycolmap 提取 SIFT 特征。不超过 60 张时做穷举匹配，更多时按 GPS 做空间匹配。然后做增量建图。少于 12 张时固定使用 EXIF 焦距，因为一个近似平面的立面只有几张照片，定不准焦距。
7. **地理配准**（`reconstruct.georegister`，纯 numpy）：
   - 用相机的右向量和相机中心的分布估计重力方向。直接平均"上"向量会因为仰拍偏差约 10°。
   - 用鲁棒的 2D 相似变换把相机中心对齐到 GPS，偏航角再按反方差和指南针融合。
   - 最后按 1.6 m 的拍摄高度确定 z。
   - 输出 ENU 坐标（x=东，y=北，z=上，单位米），以及每张照片估计出的位置和朝向。也就是说，照片的方位由 SfM 自动判断，不只依赖手机的 EXIF。
8. **输出**：`models/<地标>/<模型>.ply` 是点云，带颜色；同名 `.json` 里是注册数、点数、GPS 残差、朝向误差、每个相机的位姿和各阶段耗时。

## HTTP API

所有接口都是 JSON。时间是 unix 秒，角度是度，距离是米。字段的完整定义见 `crowd3d/api.py`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | `{"ok": true, "reconstructor": "colmap"}` |
| GET | `/api/landmarks` | `{"landmarks": [LandmarkSummary]}`：进度、扇区状态、最新全景模型 |
| POST | `/api/landmarks` | `{id?, name, lat, lon, radius_m?}` → 201 LandmarkSummary；id 重复返回 409，参数非法返回 422。不给 id 时由名称生成 slug，中文名生成 `lm-xxxxxxxx` |
| GET | `/api/landmarks/{id}` | LandmarkDetail：在 Summary 的基础上加共同完成者、每个类型/扇区的最新模型、最近 20 个作业、最近 200 张照片（含估计的位置和朝向）、阈值；不存在返回 404 |
| POST | `/api/photos` | multipart：`files`（1–50 个）、`user_id`、`landmark_id`（可选，不给时按 GPS 自动匹配地标）→ `{"results": [UploadResult], "triggered_jobs": [JobInfo]}`。校验失败放在每个文件的结果里，不作为 HTTP 错误返回 |
| POST | `/api/landmarks/{id}/rebuild` | `{kind: "face"\|"full", sector?}` → 202 JobInfo；照片数没达到最低要求返回 409，kind 或 sector 非法返回 422 |
| GET | `/api/jobs/{job_id}` | JobInfo（status 取值：queued / running / done / failed） |
| GET | `/files/models/<地标>/<模型>.ply` | 点云文件 |
| GET | `/`、`/landmark.html?id=…` | 网页（`web/` 目录） |

## 项目结构

```
crowd3d/
  config.py       Settings：所有阈值和数据目录
  geo.py          ENU 局部坐标、方位、距离
  exif.py         读写 EXIF（GPS、指南针、焦距、机型）
  records.py      Landmark / PhotoRecord / Job / ModelRecord
  validation.py   照片校验（拒绝码、警告码、模糊度、dHash）
  coverage.py     扇区、覆盖状态、作业规划（纯函数）
  store.py        SQLite 存储（线程安全）
  jobs.py         JobRunner：后台作业线程
  ingest.py       单个上传：校验 → 入库 → 保存原图和处理图
  reconstruct.py  ColmapReconstructor、地理配准、FakeReconstructor
  synthetic.py    光线投射渲染的合成建筑和照片（含 6 类坏照片）
  api.py          FastAPI 应用和命令行入口
web/              地图首页、地标页（罗盘、上传、排行、作业、模型、三维查看器）、node 单元测试
scripts/demo.py   分阶段上传的演示脚本
tests/            pytest；test_e2e.py 是端到端验收（slow），test_web.py 调用 node 测试
docs/             ARCHITECTURE.md
```

数据目录（`Settings.data_dir`，默认是 `./data`）：

```
photos/<地标>/<照片>.jpg       原图（只保存通过校验的照片）
photos/<地标>/<照片>.proc.jpg  处理图：已按 EXIF 方向转正，长边 ≤ 1600，EXIF 只保留焦距和机型
models/<地标>/<模型>.ply/.json 模型和元数据
work/<作业>/                   COLMAP 工作目录
crowd3d.sqlite3
```

不合格的上传只在数据库里留下记录（原因、位置），不保存图片。文件名一律用 uuid，不使用客户端传来的文件名。

## 配置

`crowd3d/config.py` 的 `Settings`。命令行只开放了数据目录和重建器两项，其余字段在代码里构造 `Settings(...)` 时修改。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `data_dir` | `$CROWD3D_DATA_DIR` 或 `data` | 数据目录 |
| `num_sectors` | 8 | 扇区数，0 号以正北为中心，顺时针编号 |
| `face_min_photos` | 4 | 一个扇区有多少张合格照片才生成立面模型 |
| `face_rebuild_growth` | 3 | 立面模型每多几张照片重建一次 |
| `face_margin_deg` | 10 | 立面作业额外使用相邻扇区多少度以内的照片 |
| `full_min_photos` / `full_min_sectors` | 24 / 6 | 生成全景模型需要的照片总数和就绪扇区数 |
| `full_rebuild_growth` | 8 | 全景模型每多几张照片重建一次 |
| `default_radius_m` | 150 | 新建地标的默认打卡半径 |
| `max_heading_offset_deg` | 60 | 指南针朝向偏离地标方向的上限 |
| `min_image_side_px` | 480 | 照片短边的最小像素数 |
| `blur_min_laplacian_var` | 30 | 模糊度阈值（在长边 ≤ 800 px 的灰度图上计算） |
| `duplicate_max_hamming` | 4 | dHash 距离不超过此值即判为重复 |
| `max_upload_bytes` | 25 MB | 单个文件的大小上限 |
| `processing_max_side_px` | 1600 | 送入 SfM 的处理图长边上限 |
| `reconstructor` | `$CROWD3D_RECONSTRUCTOR` 或 `colmap` | `colmap` 或 `fake` |
| `camera_height_m` | 1.6 | 假定的拍摄高度（手机的 GPS 高度太不准，不用） |
| `inline_jobs` | False | 在请求内同步执行作业（测试用） |

## 测试

分四层：合成数据上的自动化测试（每张照片的真实位姿都已知）、真实 COLMAP 的端到端验收、Web 的 node 测试，以及在冻结测试集上跑的基准（含 10 栋著名建筑的真实照片）。

- `python -m pytest -q -m "not slow"`：474 个快速测试，约 90 秒。覆盖地理计算、EXIF、校验、规划、存储、作业、上传、API、发布质量门（`qa.py`）、风格化和它的配准门与诚实检查（`stylize.py`）、合成数据、基准和测试集脚本的纯函数（`tests/test_benchmark.py`、`tests/test_eval_real.py`、`tests/test_materialize_testset.py`、`tests/test_fetch_commons.py`，都不需要真实照片），以及 Web 的 node 测试。
- `python -m pytest -q -m slow`：13 个真实 COLMAP 测试。包括单个立面、全景、空间匹配、焦距先验、照片顺序不影响结果，以及 `tests/test_e2e.py`。端到端测试用 640x480 的合成照片，8 面各 4 张，外加 6 张坏照片，通过 HTTP API 上传，按以下门槛验收：
  - 先上传北面，只触发一个立面作业，得到已地理配准的立面模型；
  - 再上传其余照片，全景模型要满足：注册率 ≥ 80%，已地理配准，GPS 残差 < 5 m，朝向误差 < 10°，≥ 80% 的相机离真值 < 5 m，≥ 1000 个点，点云水平中位数离原点 < 6 m；
  - 坏照片逐一被拒，原因与预期一致；
  - 这些门槛只适用于合成数据的小噪声（GPS σ 2 m、罗盘 σ 8°）。
- 整个 `python -m pytest -q`：487 个，2026-10-24 在提交 1d6055b 上全部通过，用时 250 s（4 核 CPU）。
- `node --test 'web/tests/*.test.mjs'`：80 个，79 个通过，1 个跳过（用 Khronos glTF 校验器检查风格化 GLB，需要设 `GLTF_VALIDATOR` 和 `STYLIZED_GLB` 才跑）。覆盖罗盘、API 封装和中文文案、模型列表、作业轮询、三维取景、风格化查看器和画风主题的校验，以及用假 DOM 跑的首页和地标页。
- **基准**（`scripts/benchmark.py`，不在 pytest 里，要先把照片取到仓库外）：同一批冻结的照片按真实用户的方式走完整条产品链路，记录发布、配准、风格化和诚实检查的指标，并和参考答案、OSM 轮廓比较。三个测试集：`synthetic-v1`（合成，几分钟）、`mdx-two-v1`（早期两栋建筑的 49 张真实 GPS 照片）、`commons-ten-v1`（10 栋著名建筑的 1,045 张 Wikimedia Commons 照片，每张自己的 EXIF 都带 GPS）。用法、指标含义和噪声基线见 [docs/BENCHMARK.md](docs/BENCHMARK.md)，结果和结论见 [docs/REAL_WORLD_EVAL.md](docs/REAL_WORLD_EVAL.md)，全部运行的汇总见 [eval/results/LEADERBOARD.md](eval/results/LEADERBOARD.md)。
  - 当前结果（核心五栋：洗礼堂、亚琛、勃兰登堡门、比萨斜塔、圣母教堂）：修复前风格化配准门放过了 51 个模型里的 31 个（0.608），可核对的 28 个里 19 个其实放错了；修复后 0 个过门（0 个错，但真实照片一面墙都不上色）。另外五栋还没跑。
  - 大约每 7 张 Commons 照片只有 1 张带 GPS（各建筑中位数 15.3%）；带 GPS 的照片里约 2/5 还带罗盘。
  - 照片各有许可，只有元数据（标题、作者、许可、坐标、哈希、数字结果）进仓库；照片、点云、GLB 和截图都不进仓库。

## 原型的局限

- **真实照片只测了五栋建筑。** `commons-ten-v1` 有 10 栋，但只跑了核心五栋，每栋最多 2 次，照片多的建筑还撞到了每栋 1 小时的上限。发布质量门和风格化配准门的阈值是在这几栋建筑上定的，属于样本内结果。见 [docs/REAL_WORLD_EVAL.md](docs/REAL_WORLD_EVAL.md)。
- **真实照片的风格化模型目前全是素模。** 配准门要求相机位置误差 ≤ 1.5 m；拍摄者离墙 40–90 m 时，只靠 OSM 轮廓和手机 GPS 证明不了这么准，所以五栋真实建筑一面墙都不上色。要恢复上色，需要不靠轮廓的偏航和尺度来源（[docs/VISUAL_LOCALIZATION.md](docs/VISUAL_LOCALIZATION.md) 的视觉定位，只有设计），或者产品决定放宽标准。
- **"0 个错误通过"只对已记录的运行成立。** 最后一轮验证发现：配准报出的偏航不确定度很大或缺失时，门会接受转错 44–180° 的对称别名解（对抗用例可复现，真实模型没碰到）；有几处决定上色的检查没有回归测试；`scripts/stylize_model.py --snap-from` 复用旧吸附时跳过新检查。三项都还没修。
- **全景模型在真实建筑上一次都没发布过**，每栋却要花 15–24 分钟反复尝试。游客只拍"好看的那一面"，很多方向几乎没有照片。
- **采集端**：只收相机原图：没有 EXIF 判 `no_exif`，EXIF 里没有 GPS 判 `no_gps`（截图、修图软件或聊天软件转存的副本通常两样都没有）。手机浏览器从相册选图时，可能会去掉 GPS（Android 上很常见）；Commons 上也只有约 1/7 的照片带 GPS。不支持 HEIC，因为没有安装 HEIF 插件。磁北朝向只给出警告，没有做磁偏角修正。
- **几何**：重力方向的估计假设手机竖直拿、地面大致水平，单个立面可能残留几度的倾斜（勃兰登堡门的模型相对参考中位倾斜 3.4°，原因没查清）。无法地理配准时，回退方案的尺度比较粗略。重建输出稀疏点云；风格化模型是 OSM 轮廓拉伸的低多边形体块：斜塔不斜，勃兰登堡门是实心盒子，配准门没过时不用点云定高度：有 OSM 高度就按 OSM，没有就用 15 m 的默认高度（洗礼堂、亚琛），屋顶都是平的（穹顶、尖顶都没有）。没有稠密重建和 3DGS。
- **重建**：只用 CPU，单线程串行。一栋 100–150 张照片的建筑在 4 核 CPU 上要 45–50 分钟。每次重建都从零开始，没有复用特征和匹配，也没有增量注册。
- **服务**：单机 SQLite 加本地文件，一个数据目录只能由一个服务进程使用。没有账号、鉴权、限流和内容审核，`user_id` 只是一个表单字段。单次请求最多 50 × 25 MB，超出时在写盘前返回 413。照片的估计位姿（已过质检、未被标记的）和模型点云对所有人公开。进程崩溃中断的作业最多重试 2 次（`max_job_attempts`）。`work/` 目录在作业结束后自动清理（`keep_work_dirs` 可保留）。
- **基准**：`scripts/benchmark.py` 的默认数据目录写死成开发机上的一个临时目录，在别的机器上要设 `CROWD3D_TESTSET_ROOT`、`CROWD3D_BENCH_DIR` 或每次传参数；新重建器要改代码注册；风格化的参数还不能通过配置调。
- **网页**：依赖 CDN 和 OSM 公共瓦片。立面模型的相机叠加只画本扇区的照片。

## 下一步

1. 用真实建筑做一次小规模实地采集，按 ARCHITECTURE.md 的质量指标重新校准阈值，包括 GPS σ、模糊度和重复判定。
2. 采集端改为页面内拍摄：同时记录 GPS 精度、指南针和重力方向（DeviceOrientation），并提示"还缺哪几面"。这样可以绕开相册丢失 GPS 的问题，也能提高重力和朝向的先验。
3. 重建升级：每个地标持久保存一个特征和匹配数据库，支持增量注册；使用 GPU 特征和匹配（如 ALIKED + LightGlue）；做稠密重建或 3D Gaussian Splatting 用于展示；在现有的发布质量门之上，加上"新版本模型不比旧版本差"的比较。
4. 平台：改用 Postgres + 对象存储，加作业队列和多个 worker，加账号、限流、人脸和车牌打码、隐私收紧，以及反作弊。

各阶段的路线图见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 第 10 节。
