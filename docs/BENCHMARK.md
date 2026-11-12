# 基准测试：怎么测、怎么比、怎么加新东西

本文是基准测试（`scripts/benchmark.py`）的使用说明，写给要跑测试、看结果、或者想换一种算法来比一比的人。

- 测出来的结果和结论在 [REAL_WORLD_EVAL.md](REAL_WORLD_EVAL.md)；外部评审采纳计划的最终评测（基线 B0 对冻结候选 F，dev 和留出集）在 [REVIEW_ADOPTION.md](REVIEW_ADOPTION.md)。
- 所有运行的汇总表在 [eval/results/LEADERBOARD.md](../eval/results/LEADERBOARD.md)，由脚本生成。
- 结果文件每个字段的含义在 [eval/results/README.md](../eval/results/README.md)。

## 0. 先说清楚它是什么

**一句话**：把一批固定的照片按真实用户的方式上传给产品，自动记下产品做出了什么、做得对不对，然后和上一次比。

- **固定**：照片清单写死在仓库里（`eval/testsets/<id>/testset.json`），每张照片都有 sha256。清单发布后不再改；要改就发一个新版本号。今天跑的和三个月后跑的是同一批照片，结果才能比。
- **按真实用户的方式**：照片通过产品自己的上传、校验、分扇区、作业规划、COLMAP 重建、地理配准、质量门和风格化，代码和默认参数都和产品相同。只上传自己的 EXIF 里带 GPS 的照片，和产品的上传规则一致。
- **对不对**：有参考答案的地方（合成数据的真值、MegaDepth-X 数据集的参考位姿、OpenStreetMap 的建筑轮廓）拿来对照；没有参考答案的地方，只记"产品自己能看到的"数字。
- **和上一次比**：`compare` 逐个指标判断"真的变了"还是"运行间的随机波动"。

仓库里只放元数据（照片标题、作者、许可、坐标、哈希、数字结果），照片和由照片得到的东西（点云、GLB、截图）一律放在仓库外。

## 1. 各部分怎么连起来

```
eval/testsets/<id>/testset.json      照片清单（只有元数据）+ outlines/*.geojson（OSM 轮廓）
        │  materialize_testset.py / benchmark.py prepare
        ▼
<data-root>/<id>/<建筑>/photos/...   本地的照片（逐张核对 sha256）
        │  benchmark.py run --config <配置>
        ▼  每栋建筑一个子进程：scripts/eval_real_landmark.py --building-spec
        │  上传 → 校验 → 作业 → COLMAP → 地理配准 → 质量门 → 风格化 → 和参考比
        ▼
<work-root>/...                      数据目录、点云、GLB、俯视图（仓库外）
eval/results/<id>/<运行名>.json       数字结果（进仓库）
        │  benchmark.py compare [--criteria] / posediff / report
        ▼
对比表和通过与否；eval/results/LEADERBOARD.md
```

| 文件 | 作用 |
|---|---|
| `eval/testsets/<id>/testset.json` | 冻结的测试集。`schema_notes` 字段逐项解释了其他字段 |
| `eval/configs/<id>.json` | 一组参数：用哪个重建器、改哪些产品阈值和算法参数 |
| `eval/splits/<id>.json`、`eval/criteria/<id>.json` | 测试集的 dev / test 划分（第 2 节）；`compare --criteria` 的通过标准（6.2 节：`review-v1`、`d31-v1`） |
| `crowd3d/params.py` | 算法参数的唯一入口（`Params`），默认值就是产品出厂值 |
| `scripts/materialize_testset.py` | 按清单把 Commons 照片取到本地并逐张核对；维护者用它挑选和冻结新的 Commons 测试集 |
| `scripts/fetch_commons_landmark.py` | 礼貌地从 Wikimedia Commons 取照片和 EXIF（被上面的脚本调用） |
| `scripts/benchmark.py` | `prepare`、`run`、`compare`（`--criteria` 按通过标准判断）、`posediff`、`report`、`rescore`（不重跑 SfM，从已有运行补算照片位姿和第 7 版的地理误差、覆盖、标定、版本指标，`--restylize` 时重新风格化），以及维护者用的 `freeze-synthetic`、`freeze-mdx` |
| `crowd3d/photoposes.py` | 每张照片的拍摄位置、朝向和误差范围（第 7.7 节），`photo_pose_*` 指标用它 |
| `scripts/eval_real_landmark.py` | 单栋建筑的测试流程和**全部指标的定义**（`METRICS`、`METRIC_SCHEMA_VERSION`） |
| `scripts/stylize_model.py`、`crowd3d/stylize.py`、`crowd3d/structure_pipeline.py` | 风格化（低多边形"玩具"模型）和它的配准门；基准对每个已发布的模型各跑一次，地标有 OSM 分件快照时走结构流水线（第 7.8 节的结构），和命令行、产品的风格化作业是同一个入口 |
| `scripts/replay_map_views.py` | 持久地图第 2a 步的离线重放：从完整模型作业留下的稀疏模型切扇区视图，和立面模型比（6.9 节） |
| `eval/results/<id>/<运行名>.json` | 一次运行的结果（只有数字和路径） |
| `eval/results/LEADERBOARD.md` | `report` 生成的排行榜，不要手改 |

**外部评审采纳计划给基准加的东西**（一览，细节在各节）：

- **指标第 7 版**（7.4、7.8 节）：模型在地图上的位置错多少（`geo_*`，朝向放反的 `wrong_face_*`）、轮廓上有多少米被模型支持（`cov_*`，建筑局部和地理两个坐标系，1.5 m 和 5 m；主要指标是 `cov_local_supported_m_1p5`）、地理锚点的 σ 准不准（`calib_*`）、旧输入有没有盖掉新输入（`stale_overwrites`）、版本漂移、运行代价（以 CPU 秒为主）和结构（`struct_*`）。这是这一轮唯一一次升指标版本，之后加的指标都在第 7 版里原地填上（7.6 节）。
- **开发 / 留出划分**（第 2 节）：只在核心五栋、合成集和 mdx-two-v1 上调；留出的五栋只用事先写明的冻结提交跑（基线 B0、候选 F）。
- **通过标准 `review-v1`**（6.2 节，`compare --criteria`）：主要覆盖指标、硬守卫、噪声内的守卫、标定区间和代价上限，dev 和 test 分别判。
- 之后的 **D31**（照片资格 `not_subject` 和立面按注册张数过门）有自己的通过标准 `d31-v1`（6.2、6.11 节）：对比的是 F，照片位姿按错误率判，另有几项要读运行产物的核对。全新运行（2026-11-03/04）的结论是 FAIL：dev 通过，test 上新显示的立面墙放偏了（两条覆盖守卫）、匈牙利国会大厦的水平 σ 说小了（一条标定），四项核对都通过。
- **配置**（4.1 节）：`cache`（地标特征缓存，验证后已是默认）、`continue`（续建，验证没通过，默认关）；`--max-requests`（6.1 节）让更快的配置在同样的上传上比。
- **工具**：`rescore`（6.4 节：`--regeo`、`--restylize --structure on|off --snap-gate on|off`）、`posediff`（6.5 节）、`calibration`（6.6 节）、`scripts/replay_map_views.py`（6.9 节）。

## 2. 测试集和版本

现在有三个测试集，`schema_version` 都是 1。

| 测试集 | 建筑 | 照片 | 参考答案 | 用来做什么 | testset.json 的 sha256 |
|---|---|---|---|---|---|
| `synthetic-v1` | synthetic-ring：8 个方向各 4 张，外加 6 张故意不合格的（全部上传）；synthetic-two-sides：只有东、东南两面，各 6 张 | 38 + 12 张，程序渲染，不用下载 | 生成器的真值位姿；轮廓是生成的方盒 | 改参数后 1–2 分钟内看有没有改坏 | `f97f7ed8…7cf15` |
| `mdx-two-v1` | 佛罗伦萨圣若望洗礼堂、亚琛大教堂 | 20 + 29 张：MegaDepth-X 里自己 EXIF 带真实 GPS 的照片 | MegaDepth-X 参考模型（组件 1） | 早期两栋建筑评测的延续，便宜（约 1–2 分钟） | `306e3672…79a51` |
| `commons-ten-v1` | 10 栋著名建筑（见下表） | 1,045 张 Wikimedia Commons 真实照片，每张自己的相机 EXIF 里都有 GPS | 549 张照片在 MegaDepth-X 参考模型里有位姿；10 个 OSM 轮廓 | 主要的真实照片测试集 | `7043a4f3…d8c09` |

`commons-ten-v1` 的 10 栋建筑（数字直接取自 testset.json 的 `stats`）：

| 建筑 | 照片 | 带罗盘 | 有参考位姿 | 打卡半径 | 拍摄距离中位数 |
|---|---|---|---|---|---|
| florence-baptistery 佛罗伦萨圣若望洗礼堂 | 56 | 39 | 20 | 150 m | 49 m |
| aachen-cathedral 亚琛大教堂 | 91 | 49 | 14 | 150 m | 61 m |
| brandenburg-gate 勃兰登堡门 | 150 | 39 | 58 + 31（西面单独一个分量） | 200 m | 52 m |
| leaning-tower-pisa 比萨斜塔 | 124 | 54 | 99 | 200 m | 61 m |
| frauenkirche-dresden 德累斯顿圣母教堂 | 111 | 39 | 81 | 200 m | 72 m |
| sagrada-familia 圣家堂 | 150 | 34 | 110 | 300 m | 75 m |
| hungarian-parliament 匈牙利国会大厦 | 97 | 17 | 36 | 800 m | 248 m |
| lincoln-memorial 林肯纪念堂 | 77 | 48 | 37 | 500 m | 100 m |
| india-gate 印度门 | 99 | 48 | 41 | 300 m | 81 m |
| sydney-opera-house 悉尼歌剧院 | 90 | 40 | 22 | 800 m | 234 m |

照片怎么收集、许可、为什么只用真实 GPS，见 [REAL_WORLD_EVAL.md](REAL_WORLD_EVAL.md) 第 2 节。

**版本规则**：

- 测试集 id 发布后**不可变**。照片清单、轮廓或参考模型有任何变化，都要发一个新 id（`-v2`、`-v3`）。冻结脚本拒绝覆盖已有的 testset.json。
- 结果文件记下 testset.json 的 sha256；`compare` 发现两次运行的 sha256 不同就拒绝比较（`--force` 才比，并给出警告）。
- 测试集目录里不许有照片：`tests/test_materialize_testset.py` 会检查目录里没有 `.jpg`、`.png`、`.ply`。

**开发集和留出集**（`eval/splits/commons-ten-v1-holdout-v1.json`）：

- 开发集（dev）是 commons-ten-v1 的核心五栋（洗礼堂、亚琛、勃兰登堡门、比萨斜塔、圣母教堂），也是这个测试集里唯一跑过 SfM 的五栋；synthetic-v1 和 mdx-two-v1 整个算开发集（mdx-two-v1 的照片就在核心五栋里：按 Commons 标题，洗礼堂 20/20、亚琛 14/29）。留出集（test）是从没跑过的五栋：圣家堂、匈牙利国会大厦、林肯纪念堂、印度门、悉尼歌剧院。
- 规则：只在开发集上调；留出集只用事先写明的冻结提交和配置跑（基线 B0 和冻结的候选 F），每次留出运行都保留；因为看了留出结果而改代码或参数，那栋建筑就"烧掉"了，划分要发新版本把它标出来。
- 摄影师：按 `materialize_testset.normalize_author` 认人，14 位摄影师两边都有，拍了留出集 513 张里的 53 张；2 张没有作者信息，不算见过也不算没见过。规则要求逐张照片的指标在留出集上另报没见过的摄影师的 458 张，工具还没分开报（第 10 节）。
- 已知的污染（每次报留出结果都要附上）：结构规则是在目录里 13 个地标的 OSM 快照上定的，包括这五栋（G3 来自林肯纪念堂的台地、G10 来自印度门）；照片位姿 EXIF 位置 σ 的标定照片包括这五栋（第 7.7 节）；印度门的参考有 12/41 台相机看来放到了拱门对面，它的对错单独报，不进合计（D31 逐张核对后，参考第 2 版把 13 台标成未验证，第 7.3 节）。
- **D31 的污染**（2026-11-03）：照片资格的占比阈值 0.33、立面按张数过门的 6 张和"拟合上 GPS"的条件、印度门参考的修正，都是看了 F 在十栋上的运行（留出五栋也在内，D31 的探针 A、B）定的（50 m 是产品负责人给的数）。按上面的规则，这五栋对 D31 来说已经"烧掉"：D31 的全新运行在 test 上只是复现测量，不是留出检验。划分文件没有发新版本（是否发由负责人定），结果仍按原来的 dev / test 标签和 F 逐划分比（6.11 节）。
- 划分文件记着它划的是哪个版本的测试集（testset.json 的 sha256），每栋建筑必须正好在 dev、test 之一里，否则 `run` 和 `rescore` 拒绝。结果的 `testset.split` 记下划分的 id、路径、sha256 和两个名单；没有划分文件的测试集（synthetic-v1、mdx-two-v1）记 `null`，整个算开发集。一个测试集有多个划分文件时要用 `--split` 指定。

**OSM 分件快照**（`eval/osm/<slug>.parts.osm`，目录里 13 个地标）：从 2026-10-22 下载的 OSM API 0.6 整块快照（地标周围的一个框）裁出来，只留 `building:part` 的线和关系、轮廓文件指定的主体、它们的成员线和节点，去掉 user、uid、changeset 和节点标签；整块快照最新元素的时间、下载日期和裁剪规则写在根元素上，`structure.parse_osm` 读回来当快照元数据。裁剪工具 `scripts/trim_osm_parts.py` 只有在裁后的文件建出和整块完全相同的结构（登记表、报告、丢掉的分件、主体标签、每块积木的三角形）时才写，并在 `eval/landmarks.json` 里记 `osm_parts`：文件、sha256、来源、下载日期、最新元素、许可（ODbL 1.0）和署名。OSM 本身不改，裁剪只是摘出。有快照的建筑（mdx-two-v1 和 commons-ten-v1 的全部建筑），快照的 sha256 进 `inputs_sha256`（第 6.1 节），风格化这一步用它做体块（第 7.8 节的结构）。

## 3. 把照片取到本地（物化）

物化后的目录结构是 `<data-root>/<测试集>/<建筑>/<照片文件>`，参考模型在 `<data-root>/<测试集>/<建筑>/gt/<分量>/sparse` 或 `<data-root>/mdx/<建筑>/<分量>/sparse`。

> **注意**：`--data-root` 的默认值是环境变量 `$CROWD3D_TESTSET_ROOT`；没设时，脚本里写死的是开发这台机器上的一个临时目录（`scripts/benchmark.py` 的 `_SCRATCH`）。在别的机器上，请设好这个环境变量，或者每次都写 `--data-root`。`--work-root`（默认 `$CROWD3D_BENCH_DIR`）同理。

```bash
. .venv/bin/activate
export CROWD3D_TESTSET_ROOT=/path/outside/repo/testsets
export CROWD3D_BENCH_DIR=/path/outside/repo/bench

# 合成集：本地生成，不下载（逐字节可复现）
python -I scripts/benchmark.py prepare --testset synthetic-v1

# mdx-two-v1：链接到本地解压好的 MegaDepth-X 场景，或者只流式下载清单里的文件
python -I scripts/benchmark.py prepare --testset mdx-two-v1 \
    --mdx-local florence-baptistery=<解压出的 Baptistry_(Florence)/1> aachen-cathedral=<Aachen_Cathedral/1>

# commons-ten-v1：按清单从 Commons 取照片，逐张核对；--gt 同时下载并核对参考模型
python -I scripts/materialize_testset.py commons-ten-v1 --cache-dir "$CROWD3D_TESTSET_ROOT" --gt
```

- **commons-ten-v1 的代价**（收集者实测，从空目录开始）：约 1,076 次请求、约 1 小时（3,499 s），其中 31 次被回 429（全部按要求等待）；1,045 张全部和清单的 sha256 一致。参考模型另需下载 MegaDepth-X 的稀疏模型（10 栋共约 6.4 GB；脚本只保留 `sparse/`）。
- **礼貌规则**是硬性的：固定的 User-Agent；对 Wikimedia 的请求严格串行，间隔 ≥ 1 s；遇到 429 按 `Retry-After` 等待并指数退避；每个响应都缓存在 `<cache-dir>/http-cache`，同一个 URL 不取第二次。不请求 robots.txt 禁止的 `/w/` 和 `/wiki/Special:` 页面。
- **离线**：`--offline` 只用缓存，不发任何请求。
- **完整性报告**：`<cache-dir>/<测试集>/integrity_report.json` 逐张写 ok / changed / missing。Commons 上的文件以后可能被替换或删除；`run` 默认只上传 ok 的照片（`integrity_policy` = skip），并把实际用到的照片字节记进 `inputs_sha256`，`compare` 遇到两边输入不同的建筑会单独列出、不进汇总。
- 脚本拒绝把缓存目录放进仓库（被 gitignore 的 `data/` 除外）。

## 4. 配置和参数

### 4.1 配置文件

`eval/configs/<id>.json`：

```json
{
 "config_id": "fast",
 "description": "...",
 "reconstructor": "colmap",
 "settings": {"processing_max_side_px": 1024},
 "params": {},
 "harness": {"arrival": "session", "seed": 0}
}
```

- `reconstructor`：重建器的名字（见第 5 节）。
- `settings`：改 `crowd3d/config.py` 里 `Settings` 的产品阈值，例如每面要几张照片、处理图多大。
- `params`：改 `crowd3d/params.py` 里 `Params` 的算法参数，例如 GPS 的 σ、罗盘的 σ、匹配方式。
- `harness`：测试流程本身。`arrival` 决定照片怎么到达：`session`（默认：同一作者同一天的照片作为一次上传请求，按拍摄时间先后）、`photo`（一张一张传）、`single`（一次全传）。
- 名字不在白名单里（`params.SETTINGS_NAMES`、`params.PARAM_NAMES`）或类型不对，直接报错。数据目录、重建器开关这类不该被调的 `Settings` 字段不在白名单里。
- 现有四个配置：`default.json` = 产品默认值（基线）；`fast.json` 只把处理图长边从 1600 px 降到 1024 px；`cache.json` 写明打开地标的特征和匹配缓存（`Params.feature_cache`，`crowd3d/landmark_map.py`：作业从缓存取前面作业算过的特征和验证过的图像对，只算新的；6.7 验证后已是默认值，和 `default.json` 等价）；`continue.json` 打开续建（`Params.map_continue`，持久地图第 1b 步：作业把新照片注册进它的目标当前显示的模型的已存地图，按规则从零重建；运行摘要的 `map` 记各模式的模型数、从零重建的原因和续建版本相对上一版的移动）。

### 4.2 参数机制（`crowd3d/params.py`）

- `Params` 是一个不可变的 dataclass，收齐了原来散落在 `reconstruct.py`、`validation.py`、`qa.py` 里的算法常量。`DEFAULTS` 等于产品出厂值，`tests/test_benchmark.py::test_defaults_are_the_shipped_constants` 把每一个默认值钉住。
- 各模块在**调用时**读 `params.current()`，所以同一份代码可以换一组参数跑。`activate()` 对整个进程生效（作业线程也能看到），`override()`、`using()` 只在一个作用域里生效。
- 每次运行把**完整的生效参数**（重建器、全部可调 `Settings`、全部 `Params`、harness 选项）和它的 sha256 写进结果文件的 `config.effective`，`compare` 会列出两次运行的参数差异。
- 计算参考答案（把参考模型配准到地理坐标）的那段代码固定用 `DEFAULTS`，换配置不会移动参考答案。
- **加一个新参数**：在 `Params` 里加字段，默认值必须等于现在的行为；在用到的地方读 `params.current().<字段>`；把默认值加进上面那个测试。这样旧结果仍然代表"默认配置"。
- 风格化自己的参数（`crowd3d/stylize.py` 的 `StyleParams`，包括配准门的各项门槛）**还不在配置白名单里**，要改只能改代码。改代码会改变代码指纹（见 6.1），`--resume` 不会复用旧的运行。

## 5. 加一个新的重建器

例如想试一个前馈模型、GLOMAP，或者 COLMAP 的另一种用法：

1. **实现接口**：`crowd3d/reconstruct.py` 的 `Reconstructor` 协议只要求一个 `name` 和一个方法 `reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir) -> ReconstructionResult`。输入是 `ReconPhoto` 列表（处理图路径、EXIF 的 GPS、罗盘、35 mm 焦距）；输出里要有每台相机的 ENU 位置和朝向（`cameras`）、`georegistered`、`yaw_sigma_deg`、`scale_sigma` 等，字段说明见 `ReconstructionResult` 的定义。
2. **注册**：把工厂函数加进 `crowd3d/reconstruct.py` 的 `RECONSTRUCTORS`，或者在一个会被 `crowd3d.reconstruct` 导入的模块里调用 `register_reconstructor(name, factory)`。必须这样做，是因为每栋建筑都在一个新的子进程（`python -I scripts/eval_real_landmark.py --building-spec ...`）里跑，只在你自己的进程里注册，子进程看不到。测试 `test_reconstructors_plug_in_through_the_registry` 演示了注册。
3. **写配置**：复制 `eval/configs/default.json` 为 `eval/configs/<新id>.json`，把 `"reconstructor"` 改成新名字，必要时加 `settings` 和 `params`。
4. **先跑便宜的**：`synthetic-v1`（1–2 分钟），再跑 `mdx-two-v1`，最后 `commons-ten-v1`。
5. **和基线比**：`compare` 要求两边测试集 sha256 和 `metric_schema_version` 都相同，所以要和同一测试集上、同一指标版本的默认配置结果比。现在的代码（合并了结构优先分支）在 synthetic-v1 和 mdx-two-v1 上的基线是 `v1-merged.json`；commons-ten-v1 还只有 `v1-after-fixes.json`（合并前的代码，见 eval/results/README.md 的 v1-merged 一节）。

两个限制：

- 相对位姿误差（`gt_rotation_error_median_deg` 等）要用到 COLMAP 格式的模型；没有的话只能比相机位置。
- 风格化和全部 `style_*` 指标要读作业目录里的 `sparse/0` 和 `images/`（COLMAP 格式）。新重建器不写这两样，风格化会记为"skipped (no kept work dir)"，`style_*` 指标是空的。

## 6. 跑、比、出排行榜

### 6.1 跑（`run`）

```bash
python -I scripts/benchmark.py run --testset synthetic-v1 --config default --name my-change-1
python -I scripts/benchmark.py run --testset commons-ten-v1 --config default --name my-change-1 \
    --buildings florence-baptistery aachen-cathedral --repeat 2
```

- 结果写到 `eval/results/<测试集>/<运行名>.json`；同名文件已存在时要加 `--overwrite`。
- `--buildings`：只跑一部分建筑。`compare` 只在两边都有的建筑上比。
- `--repeat N`：每栋建筑跑 N 次（SfM 也重跑）。记录的指标是 N 次的中位数（计数可能是 x.5），另外记下每个指标的极差（见第 8 节）。
- `--time-cap`：每栋建筑的硬上限，默认 3600 s。到 80%（2880 s）后不再开始新的上传请求，结果里 `truncated = true`，意思是"用户上传到这里为止"。照片多的建筑（勃兰登堡门、圣母教堂）在 4 核 CPU 上会撞到这个上限。
- 风格化有自己的时间预算（`STYLE_TIME_CAP_S` = 1800 s，从风格化开始算），没来得及做的模型记在 `style_incomplete`。`--no-stylize` 关掉风格化，`style_*` 指标为空。产品自己的风格化作业（`Settings.stylize_published`）在测试流程里是关的：基准在风格化这一步自己做，所以作业时间和作业 CPU 不含风格化。
- `--max-requests N` 或 `SLUG=N`（可写多个，某栋建筑自己的值优先）：每栋建筑只上传到达计划的前 N 个请求。用来和一次较慢的运行比：取它实际做完的请求数（结果 `details.requests`，例如 v1-after-fixes 的勃兰登堡门 104、圣母教堂 31），更快的配置就在同样的上传上比质量，而不是多传了照片。上限记在运行键和 `provenance.max_requests` 里，摘要里另有 `requests_planned`。
- 用了特征缓存的运行在 `details.cache` 里记下：作业数、特征和图像对的命中数与命中率、在缓存里花的秒数、缓存里的行数和大小（MB，每张照片的 MB）；每个模型的 `cache_stats`。量完以后删掉缓存文件，保存的地图（`data/maps/<地标>/models/`）留着。
- `--resume`：中断之后（例如容器重启）复用 `--work-root` 里已经跑完的建筑。只有运行键完全相同才复用：代码指纹（`crowd3d/`、`scripts/`、`eval/configs/` 的 sha256）、配置、测试集、实际输入和选项。改了任何代码，所有建筑都会重跑。
- 实际输入 `inputs_sha256`：用到的照片和它们的 sha256、轮廓、参考模型各文件的状态、OSM 分件快照的 sha256（有快照时；没有快照的合成集和以前记的一样）。参考答案已经缓存时，参考模型的状态取建缓存时记下的那份，所以缓存建好后删掉参考模型，`inputs_sha256` 不变。
- 磁盘守卫：`--work-root` 所在的盘剩余不到 3 GiB 时，不开始新的建筑运行，整个命令停下（`--resume` 能复用的已完成运行不受影响）。已经跑完的建筑留着，腾出空间后加 `--resume` 接着跑。
- 结果的 `testset.split` 记下测试集的开发 / 留出划分（第 2 节）。
- `--note`：一段说明，存进结果，也显示在排行榜上。写清楚这次运行和别的运行有什么不同、有没有跳过的建筑。
- 每栋建筑的数据目录、`summary.json`、`summary.md`、俯视图 `topdown.png`、风格化的 GLB 都在 `--work-root/<测试集>/<运行名>/<建筑>/` 下，不进仓库。
- 结果里的 `provenance` 记下 git 提交、有没有未提交的改动、Python/pycolmap/OpenCV/numpy/Pillow 的版本、CPU 数、开始时间、总用时、命令行，以及重建输出版本 `recon_output_version`（见 6.4）。
- 参考答案（第 7.3 节的参考位姿和"轮廓校正参考"）每栋建筑只算一次，缓存在 `--reference-cache`（默认 `<work-root>/<测试集>/references/<建筑>.json`），`run` 和 `rescore` 共用。缓存键是测试集 sha256（照片、参考模型的图像对应和稀疏文件哈希、轮廓）加这栋建筑的参考版本（`reference_version`：默认 `REFERENCE_VERSION`，`REFERENCE_FIXES` 里改过参考的建筑有自己的版本，所以改一栋的参考不会让别的建筑的缓存失效，见第 7.3 节），不含测试流程版本：只改指标不会让缓存失效。有缓存的建筑不再读参考模型（MegaDepth-X 的 sparse），所以缓存建好后参考模型可以删掉。缓存里是参考相机（ENU 位置和朝向，全精度）、各分量的报告、校正后的参考点（只取通过验证的分量，每栋最多 100,000 个，厘米精度）、参考自己的误差 `sigma_ref`（偏航 1°、尺度 2%；合成集的生成真值为 0），以及 `gt_model_check` 用的参考位姿。

### 6.2 比（`compare`）

```bash
python -I scripts/benchmark.py compare eval/results/synthetic-v1/v1-merged.json \
    eval/results/synthetic-v1/my-change-1.json [--noise <带噪声块的结果>] [--json out.json] [--fail-on-worse] \
    [--criteria review-v1|d31-v1|d30-v1] [--exclude-group style ...]
```

- 两边测试集 sha256 或指标版本不同：拒绝（`--force` 才比）。
- 某栋建筑两边的 `inputs_sha256` 不同（有照片变了或缺了）：单独列出，不进汇总。分件快照进指纹之前记的结果，有快照的建筑和新运行比都会显示输入不同；用 `rescore` 补算一遍，结果就按现在的方式记指纹。
- 每个指标的差值超过 **max(该指标的最小变化量, 相对下限 × 旧值, `--noise-factor` × 记录的运行间极差)** 才算"显著"。极差取自两个结果本身、`--noise` 指定的文件、同一测试集下所有带噪声块的结果中最大的那个，**但只用指标版本相同的结果**。
- 方向来自指标定义：↑ 越大越好、↓ 越小越好、"-" 只是背景信息（只报"变了"）。显著变差的个数和显著变好的个数单独统计，`--fail-on-worse` 在有显著变差时返回 1，可以接进 CI。
- 一个指标一边有、一边没有（例如某次运行多发布了一个模型）也算显著，除非重复运行记录过它本来就时有时无。
- 风格化被时间预算截断的建筑，`style_*` 指标记为 `incomplete`，不算变化。
- `--exclude-group <组> ...`：这些组的指标（组名见指标定义的 `group`，例如 `style`、`struct`、`runtime`）不进表格、不进显著变化的计数、也不进下面的通过标准。用在两边有一边的某一组不可比的时候，例如基线是旧代码风格化的：`--exclude-group style`。
- **划分**：每栋建筑属于结果记录的划分（`testset.split`）里的 dev 或 test，没记划分的结果全算 dev。同一栋建筑在 A 里是 dev、在 B 里是 test：拒绝（`--force` 才比）。两边都有 test 建筑时，除了总的汇总，还按 dev、test 各汇总一次（`splits`）；只有一个划分时它的汇总就是总的汇总。分划分汇总的噪声取这些建筑各自极差的上界（求和的指标取极差之和，其余取最大的极差），因为记录的汇总极差是按噪声运行自己的那组建筑算的。
- **通过标准**（`--criteria review-v1`，或一个文件路径）：`eval/criteria/<id>.json` 里每条规则对每个划分给出 PASS、FAIL、INSUFFICIENT（缺数据或模型不够，不算通过）、REPORT（只报告）或 EXCLUDED（组被去掉），最后给一个结论：有 FAIL 就是 FAIL，否则有 INSUFFICIENT 就是 INSUFFICIENT，否则 PASS。带 `--criteria` 时结论不是 PASS 就返回 1。规则的要求：
  - `better`：这个划分的汇总显著变好（上面的显著性规则）；`not_worse`：汇总没有显著变差；`not_worse_per_building`：汇总和每一栋输入相同的建筑都没有显著变差。
  - `max_ratio`：B 的汇总 ≤ 倍数 × A 的汇总（差值不超过指标的最小变化量也算过）。
  - `zero`：B 在这个划分的**每一栋**建筑上都是 0（输入不同的、只在 B 里的也算），重复运行的建筑看**每一次重复**（取最大的，不看中位数：三次里有一次违规，中位数还是 0）；有建筑没有值，或者 `style_*` 指标所在建筑有一次重复的风格化被截断，就是 INSUFFICIENT。写成 `{"zero": true, "skip_without": "photo_pose_gt"}` 时，各次重复都没有 `skip_without` 那个指标的建筑（例如没有参考位姿）不判，在这一行里点名留出；别的建筑没有值照样是 INSUFFICIENT，一栋都不剩也是 INSUFFICIENT（D30 加的，`d30-v1` 的两条零门槛用它）。
  - `interval`：B 在这个划分所有建筑上合并的比例（pooled 指标的分子 / 分母）本身要到区间下限，它的 95% Wilson 区间要碰到区间上限；模型数少于 `min_n` 是 INSUFFICIENT。两头不一样严，是因为 σ 说小了（过度自信）才有害：地标页画出来的范围太小；σ 对的时候 20 个模型全在 2σ 内的概率有 39%，所以超过上限只在整个区间都超过时才算不过。`max_share`：B 合并的个数 / `of` 指标的合并个数 ≤ 给定比例，同样要 `min_n`。
  - `rate_of`（D31 加的）：一个比率，分子是规则的指标（求和的计数），分母是 `rate_of` 指的指标，在这个划分的合并值上、在每一栋输入相同的建筑上都不能变差。变差要超出抽样噪声：B − A 的 95% 区间（用两边各自的 Wilson 区间拼出的 Newcombe 混合得分区间，`benchmark.rate_worse`）整个落在变差的一边。不用运行间极差，因为同一批照片重复跑看不出抽样噪声（F 两次运行在印度门上都是 3 张里错 0 张）。A 或 B 没有分母的建筑跳过并列出；整个划分都没有就是 INSUFFICIENT。A、B 两列显示比率。
  - 角色 `secondary`、`info`、`context` 只报告，不影响结论；发布数之类的只是背景，永远不是通过的理由。报告的是比例（`pooled`、单位 fraction）时，这一行还给出 B 在这个划分上合并的分子 / 分母和 95% Wilson 区间（D30 起）。
  - **标准文件里的 `checks`**（D31 起）：要读运行产物（每个模型的位姿、每次重复的漏斗）或同时看几个结果的核对。`compare` 不判它们，只在表格下面列出它们的 id，并说明上面的结论只管规则；它们按文件里写的方法另算，整个标准的结论是：规则或核对有 FAIL 就是 FAIL，否则有 INSUFFICIENT 就是 INSUFFICIENT，否则 PASS。
- `review-v1`（外部评审采纳计划的标准，B = 冻结的候选，A = 基线 B0）：
  - 主要：`cov_local_supported_m_1p5` 在 dev 上显著变好、在 test 上不变差（负责人问题 1 的默认答复：建筑局部 1.5 m；地理 5 m 是次要，只报告，由守卫 `cov_geo_wrong_m_5` 保证不变差；地理 1.5 m 只作参考）。
  - 硬守卫，两个划分每栋都必须为 0：`style_registration_wrong`、`style_qa_failed`、`struct_incorrect`、`struct_qa_failed`、`stale_overwrites`。
  - **D29 改过这个文件**（2026-11-03，原地改，`criteria_id` 仍是 `review-v1`）：放弃上色以后，硬守卫 `style_coloured_walls_wrong`（放错的模型上了多少面照片色的墙）和它的指标一起删掉，原来是六个硬守卫，现在五个；模型放没放对仍由 `style_registration_wrong` 守着。这个指标在提交的每个 B0、F 结果里都是 0，所以已经记下的通过与否不变。改动和理由写在 `review-v1.json` 的 `decisions` 里；没有任何提交的文件记着这份标准的 sha256（`compare` 只把它打印出来），所以原地改不会让旧的比较对不上。
  - 噪声内的守卫（汇总和每一栋）：`cov_local_wrong_m_1p5`、`cov_geo_wrong_m_5`、`wrong_face_cameras_shown`、`wrong_face_models_published`、`photo_pose_wrong`、`gt_gross_shown`。单栋建筑守卫变差就拦下；覆盖只看合并的。
  - 只报告（`info`）：主要指标按份额的伴随数 `cov_local_majority_m_1p5`（主要指标数点，点越密越大，见 7.8 节），两个覆盖守卫旁边按每个显示的模型加起来的 `cov_local_wrong_sum_m_1p5`、`cov_geo_wrong_sum_m_5`（守卫按段取并集，一段有一个模型支持，别的模型画错的墙就不算）。
  - 标定：偏航、尺度、水平位置、倾斜各自 `calib_<分量>_cover2` 在 [0.85, 0.99]（按上面 `interval` 的规则），`calib_<分量>_beyond3` ≤ 5%，每个划分至少 20 个模型，否则 INSUFFICIENT。水平位置和倾斜的 σ 来自地理锚点：锚点出现之前的运行（B0、v1-after-fixes）要用 `rescore --regeo` 补出，不补就是 INSUFFICIENT。B0 的留出五栋能核对的模型只有 4 个（两次重复的中位数：圣家堂 1、匈牙利国会大厦 3），所以 test 上的标定规则现在都是 INSUFFICIENT。
  - 代价：`cpu_time_s`、`peak_rss_jobs_mb`、`request_latency_p90_s`、`photo_time_to_pose_p50_s`、`disk_run_mb` 不超过 A 的 1.25 倍；`jobs_cpu_s` 在 dev 上要显著变少（持久地图），test 上不超过 1.25 倍。用 CPU 时间不用墙钟时间，因为几次运行可能同时占着 4 个核。
  - 结构的 `struct_*` 只在有 OSM 分件快照的建筑上有值：合成集没有快照，那里的 `zero` 规则是 INSUFFICIENT。B0 是结构接进基准之前跑的（按轮廓拉伸的体块），风格化指标的含义不同：和 B0 比风格化时，对候选用 `rescore --restylize --structure off`（和接入结构之前逐字节相同的老路），或者对 B0 用 `--structure on`。只看别的部分时用 `--exclude-group struct`。只有 dev 建筑的比较（合成集、mdx-two-v1）只判 dev；commons-ten-v1 的 dev 和 test 分开跑成两个结果时，分别比两次。
- `d31-v1`（D31 的标准，B = D31 的候选，A = F；在全新运行之前写好，`d31-v1.json` 的 `decisions` 写着和 `review-v1` 的每一处不同和理由）：
  - 主要指标 `cov_local_supported_m_1p5` 在 dev 和 test 上都**不变差**就行：D31 是产品负责人的规则，不是质量上的主张，不要求 dev 变好。
  - 硬守卫（每栋、每次重复都是 0）：`review-v1` 的五个，加上 `wrong_face_models_published`（`review-v1` 里是噪声内的守卫）。印度门按参考第 2 版打分（第 7.3 节），A 和 B 都是。
  - `photo_pose_wrong` 按错误率判：`rate_of` `photo_pose_gt_determined`，合并和每一栋都不能超出抽样噪声地变差。D31 有意多显示位姿（按张数过门的模型），错的张数会跟着显示的张数涨，所以 `review-v1` 那种按张数的读法只作 `info` 行列出，并在核对 `review_v1_raw_count` 里报告（预计林肯纪念堂、印度门、圣家堂会被它标出）。
  - 噪声内的守卫（合并和每一栋）：`cov_local_wrong_m_1p5`、`cov_geo_wrong_m_5`、`wrong_face_cameras_shown`、`gt_gross_shown`。标定规则和 `review-v1` 一样。
  - 代价：`cpu_time_s` 等六项每个划分都不超过 F 的 1.25 倍（`jobs_cpu_s` 也是：`review-v1` 要求 dev 上变少，那是持久地图的主张，D31 没有）。
  - 背景：`photos_accepted`、`models_qa_rejected`（每栋被质量门拦下的模型）、发布数。
  - 核对（`checks`，不在 `compare` 里判，6.11 节）：`reference_version_2`（A、B 的印度门都是参考第 2 版）、`acceptance_equals_replay`（每栋每次重复的接收数和各拒绝原因的张数等于离线重放）、`pose_correct_pooled_rise`（十栋合并的 `photo_pose_correct` 超出噪声地变多，去掉印度门也要成立）、`count_path_models`（按张数过门的模型逐个列出，合并起来位姿对的比例 ≥ 0.745，偏航 ≥ 30° 的 0 个）、`wrong_face_reference_v1`（按参考第 1 版只许印度门东面的模型错面，逐个列出；只报告）、`review_v1_raw_count`（只报告）。

- `d30-v1`（D30 视觉定位的标准，B = D31 全新运行的 `rescore --localize`，A = 同一提交上不加 `--localize` 的补算；在 B 跑之前写好，6.12 节）：
  - 没有主要指标。硬守卫（每栋、每次重复都是 0，也就是两个划分合起来是 0）：`photo_pose_loc_wrong`（定位出的、有参考位姿、显示错的张数）、`photo_pose_loc_wrong_face`。没有参考位姿的建筑（test 上的悉尼歌剧院，`photo_pose_gt` 为空）没法判，点名留出（`skip_without`）；不留出的话，test 上不管 B 怎样都是 INSUFFICIENT，`localize_photos` 永远打不开。这一处是在任何 B 运行之前改的（第九步的评审）。
  - `photo_pose_wrong_share` 合并和每一栋都不比 A 差（配对比较用 `--noise-factor 0`，门槛是最小变化量 0.01）。
  - 代价：`photo_pose_loc_cpu_s` ≤ 0.25 × `cpu_time_s`（每个划分，只看 B）。
  - 只报告：`photo_pose_loc_uncertainty_coverage`（附 Wilson 区间）、`photo_pose_correct_rate_testset` 的提高，以及定位的张数、试过的张数、每张的 CPU。
  - 核对（6.12 节）：`paired_rescores`、`only_loc_changes`、`loc_gps_conflict`（离自己 GPS 太远的定位位姿也要错 0 张）、`loc_coverage_pooled`、`synthetic_truth`、`fresh_run_display_equals_score`、`privacy_tests`、`loc_cost_live`、`loc_report`。
  - 结果（6.12 节末尾）：synthetic-v1、mdx-two-v1 PASS；commons-ten-v1 的 dev、test 都 FAIL（代价 0.267、0.665），`fresh_run_display_equals_score` 按写下的做法也不过，所以 `localize_photos` 保持关闭。代价的分母是前一天 D31 运行的 CPU，分子是第二天测的，这段时间里同样的工作要多用约 1.5 倍 CPU（6.12 节"代价的分子和分母不是同一台机器测的"）：dev 只超了 7%，落在这个漂移之内，这一条 FAIL 不稳；test 换算后约 0.41–0.46，仍然不过。

### 6.3 出排行榜（`report`）

```bash
python -I scripts/benchmark.py report       # 重新生成 eval/results/LEADERBOARD.md
```

排行榜列出全部运行的汇总值和说明、每个配置最新一次运行的分建筑结果、评审指标一节、照片位姿一节（每次带 `photo_pose_*` 的运行，分建筑列出）、各噪声运行的极差，以及完整的指标定义。它是生成的文件，不要手改。

- **划分列**：汇总表每行标出划分（dev / test，结果没记划分的全算 dev）；dev 和 test 都有建筑跑过的结果分两行，各自只汇总自己那一组建筑。分建筑的表也标出每栋的划分。
- **评审指标**（指标第 7 版起的运行，按划分汇总）：`review-v1` 看的那些数（主要覆盖、地理 5 m 覆盖、两个覆盖错误、地理相机误差、朝向放反、照片位姿正确率、偏航和尺度的标定、硬守卫、CPU 时间、请求延迟、峰值内存、磁盘）。通过与否要用 `compare --criteria` 和基线比，这一节只列数。

### 6.4 补算（`rescore`）

照片位姿指标（第 7.7 节）和第 7.8 节的地理误差、覆盖、标定、版本指标只看一次运行留下的东西：数据目录里的 SQLite（照片、作业、已发布模型和每台相机的位姿）、PLY、保留的作业目录里的 `sparse/0`、`summary.json` 里的照片行、上传请求记录和风格化接受的轮廓吸附。所以已经跑过的运行不用重跑 SfM，就能补算：

```bash
python -I scripts/benchmark.py rescore --testset commons-ten-v1 \
    --from eval/results/commons-ten-v1/v1-after-fixes.json --name v1-poses \
    [--restylize [--structure on|off] [--snap-gate on|off] [--replay <配置>]] [--regeo] [--localize]
```

- `--restylize`：先用现在的代码把运行保留下来的每个已发布模型重新风格化一遍（和 `run` 里一样，同一份参考答案），`style_*` 指标和照片位姿都用新的风格化结果；风格化的 GLB 和每次重复的 `style.json` 写在 `<work-root>/<测试集>/<运行名>/<建筑>/<重复>/`，源运行目录不动。同样的代码、同样的参考，重新风格化得到的记录和运行时的一样（时间和路径除外；`tests/test_eval_real.py` 的慢测试核对）。
- `--structure on|off`（只和 `--restylize` 一起用，默认 `on`）：`on` 和 `run` 一样，地标有 OSM 分件快照时风格化走 `crowd3d.structure_pipeline`（带点的观测轨迹，`struct_*` 指标）；`off` 走只用轮廓的老路（不带轨迹，`struct_*` 为空），是结构这一项的对照：同样的代码下它的 GLB 和接入结构流水线之前逐字节相同。结果的 `provenance.structure` 记下用的是哪一种。
- `--snap-gate on|off`（只和 `--restylize` 一起用，默认按 `StyleParams` 的默认值，现在是 `on`；`off` 是 #73 修复之前的门）：任务 #73 的三个吸附门开关（`stylize.SNAP_GATE_73`）一起打开或关上：偏航窗口用地理锚点的偏航 σ（`snap_use_anchor`，作业目录也按锚点的变换放到 ENU）、对称周期截短了窗口或者模型没报偏航 σ 就拒绝（`snap_refuse_clipped_window`）、吸附后相机到自己 GPS 的中位距离超过 max(20 m, 3 × 锚点在相机处的水平 σ) 就拒绝（`snap_max_gps_median_m`）。锚点出现之前的模型 JSON 没有锚点：加 `--regeo` 时用重跑出的锚点，不加时偏航 σ 照旧用 `yaw_sigma_deg`。结果的 `provenance.snap_gate` 记下三个开关的值。
- 重建输出版本：`crowd3d/reconstruct.py` 的 `RECON_OUTPUT_VERSION` 标记一次重建作业对同样的上传留下什么（稀疏模型、PLY、模型 JSON、质量门结论）。能改变这些输出的改动（SfM 选项、选模型、地理配准、相机检查、导出、质量门、作业规划、照片接收、`Params`/`Settings` 默认值）都要把它加 1；`tests/test_recon_golden.py` 让合成照片走产品的路（`ingest_upload` 接收、作业规划、`JobRunner` 跑作业），COLMAP 的特征、匹配和建图换成一个固定的合成稀疏模型，核对接收后的照片记录（含 EXIF 的 35 mm 焦距）和处理过的副本、作业的照片顺序、传给 COLMAP 的选项和数据库里的相机（焦距先验）、以及之后的每一步，输出变了而版本没变就失败。只给模型 JSON 加键、原有的键和值都不变的改动不用加版本：测试会要求重新生成一次 golden 把新键记下来（`CROWD3D_REGEN_GOLDEN=1`；原有的值变了时拒绝）。`rescore` 拒绝补算版本不同的运行（`--allow-recon-diff` 才算，结果记下它算的是哪个版本）。所以只改风格化或指标的提交之后，旧运行的产物还能接着补算。这个版本出现之前的结果按提交对应（`LEGACY_RECON_OUTPUT_VERSION`）：v1-merged（0a49b22）和 v1-after-fixes（1d6055b、3bbb0dd）是版本 1，v0 的运行不是。版本 2 从 D31 开始（fcb4fbd 的照片资格、0ebbcc1 的立面按张数过门都改了产出）；D31 自己那四个提交（fcb4fbd…09f7621）上跑的运行（`d31`、`d31-dev`、`d31-test`）记的还是 1，实际是版本 2 的产物，按提交对应（`MISRECORDED_RECON_OUTPUT_VERSION`，只管运行，不管补算）。版本 3 是 D34 第 6 步的 6f9e8cf（地理配准的三个开关都成了默认值，6.13 节），只存在于那一个提交，没有哪次运行是用它记录的。版本 4 是那一步评审之后：偏航裁决 `yaw_arbitration` 改回默认关（dev 的规则选不到它），另外两个开关不变。所以现在补算 D31、D30、F 和更早的运行都要加 `--allow-recon-diff`：算出来的是那时的地理配准和质量门的结论，不是现在的。带 `--regeo` 或 `--replay` 时，旧运行没有记录的开关按开关出现之前的行为算（`params.recorded_run`），所以版本 2 的运行照样能逐相机重现，6.13 节的重放命令现在重跑要多加一个 `--allow-recon-diff`。

- 读 `--from` 结果的 `artefacts_dir`，对每栋建筑的每次重复（r1、r2……）各算一次，记录中位数，和 `run` 一样。重算的是 `photo_pose_*`、第 7.8 节的 `geo_*`、`wrong_face_*`、`cov_*`、`calib_*`、`stale_overwrites`、`version_drift_*`，以及请求记录里有的运行代价（请求延迟；第 7 版的运行还记了作业 CPU 时间、峰值内存和每张照片多久有位姿，旧运行没记，这几个为空）；其余指标（包括 `cpu_time_s`、`disk_run_mb`、`wall_time_s`）从源结果原样拷过来，旧版本的结果里没有的就是空。`details` 按现在的代码重新整理（每个模型多了 `geo` 和 `coverage`）。结果里 `provenance.rescored_from` 记下源结果的名字、路径和 sha256。
- `inputs_sha256` 按现在 `run` 的方式重新算（含分件快照），所以补算过的旧运行能和新运行在同样的输入上比；本地的照片、轮廓或参考和源运行用的不一样时（新旧两种指纹都对不上源结果），会警告，并把 `inputs_sha256` 记成一个以 `unmatched:` 开头、任何运行的指纹都不会等于它的标记（`inputs_unmatched` 记下源结果的指纹和本地的指纹）：这些产物不是今天的输入做出来的，`compare` 把这栋建筑列在"输入不同"里，不进汇总。`testset.split` 记现在的划分。
- 数据目录只读：SQLite 先复制到临时目录再打开（`Store` 打开时会给旧库加列）。
- 源结果里没有的建筑（commons-ten-v1 只跑过核心五栋）写 `skipped (no run in the source result)`，不打分。
- 参考答案用 6.1 的缓存（`--reference-cache`，和 `run` 共用）。缓存不存在、参考模型又不在本地的建筑没有参考答案，对错类指标为空。
- 测试集分母：每栋有 SfM 运行的建筑，`rescore` 按测试集重新列出流程本该上传的照片（文件在、有 EXIF GPS；`upload_all` 的合成集是全部），和源运行 `summary.json` 的漏斗核对（对不上会警告）。时间上限截掉、没上传的照片在 `*_testset` 指标里算失败，每次重复的 `poses.runs[]` 记下 `testset_photos`、`not_uploaded`、`not_uploaded_gt`。
- 结果文件同名已存在时要加 `--overwrite`。
- `--regeo`：用产品代码（`reconstruct.georegister_inputs`、`georegister_model`）、源运行自己的参数（`summary.json`）、数据库里照片的 GPS 和罗盘，在保留的 `work/<作业>/sparse/0` 上把每个已发布模型的地理配准重跑一遍，但地理锚点的下限用现在的（`Params` 里 `anchor_*`）。重跑出的每台相机都要在模型 JSON 的 ENU 位置 1 cm 内（`REGEO_MAX_DIFF_M`），否则这个模型不进标定；`calib_*` 用重跑出的锚点。每次重复的结果记在建筑记录的 `regeo`（已发布、重跑了、一致的模型数，最大差，没进标定的模型和原因），`provenance` 记 `regeo` 和锚点的标定（`anchor_calibration`）。锚点出现之前的运行（B0、v1-after-fixes）只有这样才有水平位置和倾斜的标定。v1-after-fixes 核心五栋 125 个已发布模型全部一致（最大差 0.0000 m），CPU 约 12 秒。
- `--replay <配置>`（D34）：回答"同样的 SfM 输出，换一组参数会发布什么"。运行里存下的每个模型（发布的和没发布的），按创建顺序，用现在的代码从它的作业保留的稀疏模型（`work/<作业>/sparse/0`）重做一遍：地理配准和相机检查（`reconstruct.georegister_inputs`、`georegister_model`，用运行自己的参数加上配置里的 `Params`）、质量门（`qa.assess`，用运行自己的设置）、"同一目标已有更新的模型"的替换规则（`jobs.superseded_by`）。重做出的运行目录写在 `<work-root>/<测试集>/<运行名>/<建筑>/<重复>/replay/`：数据库是副本，照片、作业目录和地图是指向源运行的链接，源运行目录不动。之后的打分和重新风格化都读这个目录。
  - 每个模型先用运行自己的参数重放一次。重放出的每台相机都在模型 JSON 的 1 cm 内，相机标记、点云的水平中位数、配准与否、质量门的结论和理由都和存下的一样，才换参数重做；对不上的模型（包括没有保留稀疏模型的）原样留着，列在结果里（`left_as_recorded`，带原因）。
  - "运行自己的参数"取自 `summary.json` 的 `effective_params`（`params.recorded_run`，`--regeo` 也用它）：运行之后才加进 `Params` 的开关，运行没有记录，按"这个开关出现之前"的行为算（`params.AS_BEFORE`），不跟着后来的默认值走。这样以后哪个默认值改了，旧运行照样能逐相机重现。地理锚点的下限始终用现在的。
  - 罗盘读数标的是真北还是磁北（`heading_ref`）：照片行有就用照片行的；D34 之前记录的运行，数据库的照片表没有这一列，取 `summary.json` 的 `photos` 里记的。地理配准和相机检查用的是这样取到的标记。质量门判断"哪些照片没有罗盘替它担保"时读的是照片行，取不到：`compass_other` 不是 `trusted` 的重放里，标成真北的照片也会被查"估计出来的朝向是不是背对建筑"，全新的运行不会查它们。量到的差别见 6.13 节末尾。
  - 只收 SfM 之后才读的 `Params`（`REPLAYABLE_PARAM_PREFIXES`：GPS 拟合、罗盘、偏航、重力方向、相机标记、`qa_*`）。带 `Settings`、换了重建器或者改了别的 `Params` 的配置被拒绝：重放不出它们的效果。配置 `default` 就是用运行自己的参数重放，结果要和不加 `--replay` 的补算一样。
  - 必须和 `--restylize` 一起用（存下的吸附是在原来的模型上量的），不能和 `--regeo`、`--localize` 一起用；重放出的结果不能再拿来当 `--from`。
  - 重放看不到三件事：作业在几次建图尝试里会选哪个候选；可疑的候选触发的重试（运行只保留了被选中的那个模型）；前面的模型发布与否变了以后，后来的上传会排出哪些作业。
  - 每栋建筑的记录多一个 `replay` 块，每次重复一项：模型数、重放了几个、原样留下的、变换变了的、配准等级变了的、发布状态变了的（后两样逐个列出）、重放后各等级和已发布的模型数。`provenance.replay` 记配置的 id、路径、sha256 和参数。
- `georeg`（D34，单独的子命令）：`python -I scripts/benchmark.py georeg <结果> [<结果> ...] [--split dev|test] [--json <文件>]` 把几个结果并排列成表，总体一张、每栋一张：
  - 能和参考比的模型（标定表里的行：已发布、已配准、至少 3 台没被标记而且有可信参考位姿的相机；同一个模型在两次重复里只算一次）的尺度误差（|ln 尺度| 的均方根、偏差超过 35% 的个数）、偏航误差（均方根、中位数、超过 15° 的个数）、锚点支点处的水平误差中位数；
  - 发布了几个模型、显示了几个位姿、对的和错的各几个、超出自己 2σ 范围的几个、四个分量的 2σ 覆盖率（这些是每栋两次重复的中位数再加起来，和 `compare` 的汇总一样）；
  - 按模型注册的照片数分四档（3–6、7–10、11–20、21 张以上）的尺度和偏航误差（D34 之前记录的标定行没有张数，不进任何一档）；
  - 结果是 `--replay` 做的话，再列重放改了什么（两次重复都算）。

- `--localize`（D30）：在每次重复保存的地图上跑产品的视觉定位（`eval_real_landmark.rescore_localize`：`crowd3d.localize` 的 `candidate_maps`、`eligible`、`load_sparse_maps`、`ColmapLocalizer.localize` 和它的配对选择原样调用，记录按候选地图的 `maps_key` 存），把被接受的定位当作来源 `loc` 打分（产品 `localize_photos` 打开时的显示规则）。吸附用打分时用的那些，同意按打分时的假设。这些运行没留下特征缓存，特征要重新提：先把地图照片的特征放进这次重复自己的缓存（从这栋楼的特征库里拿，库里没有才提；CPU 记在 `poses.runs[].loc.map_features`，不算指标），定位器再在这个缓存上提查询照片的特征、算每个图像对：这一步的进程 CPU 是 `photo_pose_loc_cpu_s`，重复之间不共用查询特征和图像对。特征库在 `<work-root>/<测试集>/<运行名>/<建筑>/loc-cache/`，这栋楼做完就删；每次重复的记录（库内照片 id、精确位姿，只在暂存目录、从不进仓库）写在 `<work-root>/<测试集>/<运行名>/<建筑>/<重复>/loc.json`，以后可以离线重新打分。数据目录照样只读（SQLite 复制后打开，处理图只建链接，地图只读）。每次重复的 `poses.runs[].loc` 是计数（第 7.7 节），`provenance.localize` 记下定位器、规则版本、`loc_*` 参数、特征版本和 CPU 的分法。慢测试 `tests/test_benchmark.py::test_rescore_localize_scores_a_recorded_run` 在合成方盒两面的一次真实 COLMAP 运行上核对：建模之后才上传的那张照片不加 `--localize` 时按 EXIF 显示，加了以后定位出来，离真值 1 m 内，去掉地图自己的朝向误差后朝向差 1° 内；别的数都不变；运行目录逐字节不变。

### 6.5 两次运行最后显示的位姿差多少（`posediff`）

```bash
python -I scripts/benchmark.py posediff eval/results/commons-ten-v1/<A>.json eval/results/commons-ten-v1/<B>.json \
    [--buildings <建筑> ...] [--json out.json]
```

- 每栋两边都跑过的建筑，读两边运行结束时地标页显示的位姿（`crowd3d.api.camera_estimates`：每张照片一个，已发布模型里没被标记的相机），在两边都显示了的照片上比相机中心：直接比（两边都是地理配准后的 ENU），以及把 B 用 Sim3 对齐到 A 之后比（`sim3_position_errors`，去掉粗差再拟合一次，剩下的是形状上的差别）。给出照片数、中位数 / 90 分位（米）和 Sim3 的尺度。
- 旁边列出每个结果自己第 1 次和第 2 次重复之间同样的差（有重复时）：同一配置运行间的波动。持久地图的验收（续建的最终模型和从头重建的比，中位数不超过 max(这个波动, 0.5 m)）就看这两行。
- A 和 B 各取第 1 次运行比。运行目录（`artefacts_dir` 下的 `summary.json` 和数据目录）要还在，不在的建筑写 `artefacts gone`；测试集不同的两个结果拒绝比较。截断点不同的两次运行上传的照片不一样，所以只比两边都显示的照片（`common`）。

### 6.6 标定表和下限拟合（`calibration`）

```bash
python -I scripts/benchmark.py calibration eval/results/commons-ten-v1/<补算>.json [更多结果 ...] \
    [--split dev|test] [--floors yaw_deg=..,log_scale=..,xy_m=..,tilt_deg=..] [--fit] [--json out.json]
```

- 读结果里每栋建筑的 `calibration` 行（`run` 和 `rescore` 都写），几个结果、几次重复里同一个模型（同一个 `key`）只算一次。
- 每个分量（偏航、尺度、水平位置、倾斜）按建筑和按建筑加权（每栋一票）、以及全部合在一起，列出：模型数、误差中位数（单轴分量还有带符号的中位数，即偏差，模型减参考）、σ 中位数（锐度）、\|z\| 中位数（σ 准时 0.674）、1/2/3 个 σ 内的比例（0.683/0.954/0.997）和 3 个 σ 外的比例。
- `--floors`：把这组下限和每个锚点自己的 σ 平方相加再开方，代替算分时的 σ（没有锚点的行不变）。
- `--fit`：每个分量找最小的下限，让按建筑加权的 2 个 σ 内比例达到 0.954（只有锚点自己的 σ 和误差决定一个模型从哪个下限起落进 2 个 σ，所以候选就是这些门槛值），向上取到 0.01°/0.001/0.01 m/0.01°，再按拟合的下限出一张表。
- 下限只在开发集上拟合，在任何留出集的重跑之前冻结在 `Params` 里，`anchor_calibration_id` 给这组下限起名。下限改了不用加重建输出版本：锚点的 σ 和 `calibration` 跟着新的 `anchor_calibration_id` 变，`tests/test_recon_golden.py` 允许这一种变化（要求每个 σ 正好是锚点自己的 σ 加下限），补算按现在的下限算分。锚点的算法改了（`crowd3d.anchor.ANCHOR_METHOD`，记在锚点的 `method`，没有这个键的是 1）也不用加重建输出版本：锚点能从保留的 SfM 模型重算，金样测试允许整个锚点变、其余值一个都不许变；存下的旧算法锚点用 `rescore --regeo` 按现在的算法重算。方法 2（倾斜的自助法绕东、北两轴，相机按 ENU 中心排序，输入放到 2^-30 的网格上）让同一个重建换一个 SfM 坐标系（COLMAP 每次跑的规范不同）或换一个图像顺序得到同一个倾斜 σ，相机差一点 σ 也只差一点；方法 1 下圣母教堂 r1、r2 的同一批模型倾斜 σ 差到 2.349° 对 2.418°、9.388° 对 12.237°，3–6 台相机的模拟模型差到 32°（重采样里同一台相机的几份拷贝求平均有舍入误差，`estimate_up` 把这点噪声归一化成任意方向；网格上几份拷贝的平均正好是它自己）。

**现在的下限 `dev-core5-v1`**【开发集，样本内】：commons-ten-v1 核心五栋 v1-after-fixes 的 `rescore --regeo`（125 个已发布模型全部一致），两次重复去重后 86 个核对模型（洗礼堂 3、勃兰登堡门 31、比萨斜塔 23、圣母教堂 29；亚琛 0 个），按建筑加权拟合：偏航 7.35°、log 尺度 0.352、水平位置每轴 3.1 m、倾斜每轴 1.94°。按建筑加权的 \|z\| 中位数和 1/2/3 个 σ 内、3 个 σ 外的比例：

| 分量 | 旧字段的 σ（`yaw_sigma_deg`、`scale_sigma`） | 锚点，拟合前（偏航下限 3°，其余 0） | 锚点，`dev-core5-v1` |
|---|---|---|---|
| 偏航 | 1.07；0.47/0.84/0.94；0.056 | 0.98；0.58/0.90/0.97；0.030 | 0.48；0.81/0.96/0.99；0.011 |
| 尺度 | 1.12；0.47/0.69/0.78；0.218 | 1.12；0.47/0.72/0.80；0.202 | 0.22；0.87/0.96/0.99；0.009 |
| 水平位置 | - | 1.94；0.17/0.57/0.81；0.187 | 0.68；0.70/0.97/0.99；0.009 |
| 倾斜 | - | 0.53；0.71/0.80/0.83；0.172 | 0.15；0.84/0.96/1.00；0 |

- 误差是重尾的：要让 95% 落进 2 个 σ，σ 得比让 68% 落进 1 个 σ 的大得多（偏航按建筑加权 68% 只要下限 3.9°，尺度 0.10，倾斜 0），所以拟合后中间的模型 \|z\| 偏小（σ 偏保守），尾巴还在：比萨偏航 2 个 σ 内 0.87（3 个模型在外，其中有转错面的），圣母教堂尺度 0.83、水平位置 0.86。尺度 σ 中位数 0.36（约 ±43%），这样的 σ 当不了尺度门槛；消费方在留出集标定通过之前继续用旧的 `scale_sigma`。
- 只有 4 栋建筑：去掉任何一栋重拟，偏航 5.0–8.1°、尺度 0.16–0.44、水平位置 3.0–3.1 m、倾斜 1.4–2.0°（去掉比萨偏航降到 5.0°，去掉圣母教堂尺度降到 0.16）。所以这是候选，留出集（S7）才是结论。
- 冻结后用新的默认值重跑 `rescore --regeo`，`calibration` 的表和离线拟合的表逐位相同。合成集和 mdx-two-v1（没参与拟合）在这组下限下 2 个 σ 内都是 1.0（合成集的生成真值误差小于下限，按设计偏保守）。

### 6.7 地标特征缓存的验证（持久地图第 1a 步）

`crowd3d/landmark_map.py` 的 `FeatureCache`（`Params.feature_cache`，验证通过后默认开）：作业开始前把地标缓存里这些照片的特征、这些图像对的匹配和两视图几何原样（SQLite 的行，逐字节）写进作业自己的 COLMAP 数据库，COLMAP 跳过已有的图像和图像对，只算新的；匹配完、建图前把新算的写回缓存。键：处理图的 sha256 加特征版本（pycolmap 版本、提取参数）；图像对另加 COLMAP 匹配它们时的先后（配对生成器里第一次出现的顺序）、两张图的 id 先后、两台 COLMAP 相机的签名和匹配版本。只写今天的配对会选的图像对（穷举，或 GPS 空间配对），所以缓存不会给作业多加图像对；两视图几何里带 COLMAP 自己估的相机（存的是作业里的相机 id）的图像对不进缓存。

验证（2026-10-29，worktree 钉在 031773b，配置 `cache`，`--no-stylize`，4 核 CPU，同一时间只跑一个 COLMAP）：

- **数据库逐字节相同**：重放已记录运行的作业（`ColmapReconstructor.reconstruct`，建图换成空操作），每个作业的数据库建两次，一次用缓存（按作业顺序共用一个缓存）、一次不用，比较建图读的四张表（keypoints、descriptors、matches、two_view_geometries）：合成环 9 个作业、洗礼堂 8 个、勃兰登堡门前 29 个（其中 74 张和 82 张照片的两个是 GPS 空间配对）共 46 个作业、18,156 行，**全部相同**。所以缓存不改变建图的输入；运行之间剩下的差别都来自 COLMAP 建图本身的不确定性（多线程光束法平差；不用缓存的重复运行之间也有）。
- **synthetic-v1 ×2**：和 B0（v1-merged 的第 7 版 `--regeo` 重算）比，`compare --exclude-group style struct poses` 没有显著变化；三个指标的中位数差一点（轮廓旋转 +0.0025°、参考旋转 +0.0005°、参考位置 +0.0002 m，都在噪声内），来自合成环第 1 次重复的建图；B0 新跑的两次重复之间也有同样大小的差（两面 2.98° 对 2.955°）。只有 `disk_run_mb` 显著变大（8.3 → 18.4 MB，保存的地图）。
- **mdx-two-v1 ×5**：和 B0 比没有显著变差的质量指标（`disk_run_mb` 33.3 → 39.1 MB 除外）。
- **commons-ten-v1 核心五栋 ×2**：勃兰登堡门只传前 104 个请求、圣母教堂前 31 个（`--max-requests`，v1-after-fixes 两次重复都只做到这里），其余三栋全部请求。和 v1-after-fixes 的 `--regeo` 重算比，按两次重复的噪声算有 7 处显著变差：汇总的 `models_qa_rejected` 62 → 63.5、`estimate_coverage` 0.600 → 0.590、参考旋转中位数 0.802 → 0.874°、地理偏航中位数 3.87 → 4.89°，勃兰登堡门的地理偏航 4.09 → 4.62° 和版本漂移 p90 4.82 → 5.65 m，圣母教堂的参考旋转 0.746 → 0.864°。数据库相同，这些是建图的随机性：逐个模型看，勃兰登堡门缓存运行的每个模型的偏航误差都等于某次不用缓存的运行的值（例如第 50 个请求的东立面：v1 第 1 次和缓存两次都是 6.45°，同一提交不用缓存的对照是 3.07°），两次重复的噪声带低估了这种两种结果都可能出现的模型（第 8 节）。发布数、立面数、配准率、`gt_gross_shown`、`wrong_face_*`、`cov_local_supported_m_1p5`、`stale_overwrites` 没有显著变差。
- **命中率**（核心五栋两次重复合计）：特征 89.2%（7,564 / 8,482），图像对 73.3%（98,071 / 133,838）；洗礼堂 48% / 35%，亚琛 69% / 53%，勃兰登堡门 93% / 80%，比萨 88% / 71%，圣母教堂 90% / 62%。合成集 52% / 15%、mdx 25% / 9%（作业少、照片少）。缓存每张照片 1.09–1.74 MB（核心五栋平均 1.39 MB），每栋运行在缓存里花 1–24 s。
- **CPU**：作业 CPU 秒（`jobs_cpu_s`）和同一提交、同样请求前缀、不用缓存的对照（各 1 次）比（缓存取两次重复的平均）：洗礼堂 427 → 337 s（−21%），亚琛 1,071 → 490 s（−54%），勃兰登堡门 8,996 → 2,279 s（−75%），比萨 5,851 → 2,657 s（−55%），圣母教堂 6,154 → 3,435 s（−44%），合计 22,497 → 9,197 s（**−59%**）；作业墙钟时间 7,475 → 4,602 s（−38%）；作业的峰值内存 2.9–3.2 GB → 2.7–3.0 GB。和对照比，按单次对照、两次缓存重复的噪声带也有几处"显著"变化（两个方向都有），把 v1 的两次、对照和 b0-dev 都算作不用缓存的运行时，232 个（建筑、质量指标）里缓存两次都落在它们范围外的有 10 个，其中只有勃兰登堡门的地理偏航、墙面误差 p90 和版本漂移 p90 偏同一边（就是上面那个两种结果都可能出现的模型）。v1-after-fixes 没记 CPU，而且它的作业时间是今天不用缓存对照的约两倍（洗礼堂 357 s 对 183 s，当时机器负载不同），所以成本只和同一提交的对照比：1a 的成本判据按作业 CPU 秒判（−59%），作业墙钟时间是对照的 62%（4,602 / 7,475 s），没达到设计里的 ≤ 50%。不设上限的勃兰登堡门、圣母教堂全量运行（吞吐量、首个发布模型的时间）这一步没有跑，后来在冻结候选 F 上补跑了（6.10 节）。合成集作业 CPU 比 B0 新跑少 31%（165 → 113 s），mdx 少 25%（157 → 117 s）。

所以 `Params.feature_cache` 默认打开。`eval/configs/cache.json` 留着，把这个开关写明。

### 6.8 续建的验证（持久地图第 1b 步）

`Params.map_continue`（`crowd3d/jobs.py` 的 `_map_plan` 选底模，`ColmapReconstructor._continue` 续建）：作业把新照片注册进它的目标当前显示的模型的已存地图（pycolmap 增量建图从那个模型继续，旧相机也参与光束法平差），地理配准和逐相机检查照旧；保住底模 95% 已注册照片、不可疑（`suspicious()`：超过 5% 的相机和自己的传感器对不上，或罗盘误差均值超过 20°）、看错立面的相机少于 2 台才接受，否则照常从零建图。照片数到上次从零重建的 1.5 倍、连续续建 4 次、或续建的模型没过质量门时从零重建。配置 `continue`（`eval/configs/continue.json`）打开它。

验证（2026-10-30，worktree 钉在 be264f8，配置 `continue`，`--no-stylize`，4 核 CPU，同一时间只跑一个 COLMAP；和 6.7 同一组：synthetic-v1 ×2、mdx-two-v1 ×5、核心五栋 ×2，勃兰登堡门前 104 个请求、圣母教堂前 31 个）：

- **合成集和 mdx**：一次续建都没有发生（每个目标要么只建一次，要么下一次已到 1.5 倍），和 1a 比没有显著变化，和 B0 比只有 `disk_run_mb`（保存的地图，同 1a）。
- **核心五栋**：两次重复共 304 个作业，和 1a 完全是同一批作业。每次重复 152 个作业：18 个续建模型、20 次续建被拒后从零重建（continued->scratch）、86 个从零建图的模型、28 个作业没出模型；20 次被拒全都是 `suspicious()`（6–20 台相机里 1–2 台被标记），没有续建的模型没过质量门（所以没有发生"没过门重跑一次"）。
- **硬指标**：`gt_gross_shown` 6（B0 7、1a 11），`wrong_face_cameras_shown` 3.5（4、3.5），`wrong_face_models_published` 0（0.5、0.5），`stale_overwrites` 0。
- **质量**：和 1a 一样做范围检验：167 个（建筑、质量指标）里，续建两次重复都落在 6 次从零建图的运行（v1 两次、1a 两次、同提交不用缓存的对照、b0-dev）范围外的有 31 个（缓存只有 232 个里 10 个），其中 13 个偏坏、18 个偏好。偏坏：亚琛北立面的续建版本（14 张照片）轮廓平移 9.6–10.8 → 52.4 m、轮廓残差 0.40–0.56 → 0.86 m、GPS 残差 7.6–7.8 → 8.8 m；圣母教堂参考位置误差中位数 1.10–1.14 → 1.27–1.32 m、错误覆盖长度；勃兰登堡门地理尺度误差 9–16% → 20%、罗盘误差、5 m 容差的支撑覆盖 61 → 38 m；洗礼堂轮廓旋转 1.6–2.1 → 2.7°。偏好：勃兰登堡门 1.5 m 容差的错误覆盖 45–53 → 3–5 m、支撑覆盖 8–16 → 33–35 m、地理偏航 4.0–4.7 → 2.1–2.2°、版本漂移；比萨参考旋转。和 1a 的汇总比：建筑局部 1.5 m 的支撑覆盖 133 → 157 m（主覆盖指标，变好）、错误覆盖 82 → 39 m；5 m 的支撑覆盖 216 → 195 m、地理尺度误差中位数 11.5 → 14.1%、`jobs_failed_sfm` +1（变差）。
- **版本连续性**：36 个续建版本里 20 个相对上一版在 2 m、3° 以内（56%，要求 90%）；超过 30 张照片的 20 个里 18 个在内，30 张以下的 16 个里只有 2 个。续建版本的移动中位数 1.50 m、朝向 0.67°，Sim3 对齐后 0.26 m；同一批运行里从零重建的版本只有 29% 在内（中位数 4.3 m），v1-after-fixes 全部是从零重建，28%（3.0 m，对齐后 0.87 m）。续建让版本更稳，但小照片集的地理配准每版都按 GPS 重新拟合（偏航 σ 5–7°），整体跟着动。
- **和从零重建的位姿差**（`posediff`：v1-after-fixes 第 1 次对续建第 1 次，上限是 max(v1 两次之间的差, 0.5 m)）：洗礼堂 0.05 m（两次之间 0.66 m），亚琛 4.12 m（0.22 m，上限 0.5 m；Sim3 尺度 1.22），勃兰登堡门 1.27 m（0.83 m），比萨 0（0），圣母教堂 0.78 m（2.0 m）。五栋里两栋超出。
- **成本**：作业 CPU 8,219 s，1a 9,197 s（−10.6%），不用缓存的对照 22,497 s；作业时间 4,054 s，1a 4,602 s，v1 9,410 s（43%，要求 ≤ 40%），对照 7,475 s。但这批运行时机器更快：同一批作业里只有从零建图的请求，CPU 比 1a 少 13.9%。按这个比例修正后，含续建作业的请求省 231 s，含被拒续建的请求多 831 s（先试续建、再整个从零建图），**净省为零**（反而多约 600 s）。续建作业本身：作业时间 473 s，1a 里同样的作业 913 s。续建运行的作业时间去向：完整模型 39%（从没发布过，所以从不续建）、被拒续建后的从零重建 33%、续建 5.8%。

所以续建这次没通过（质量有偏坏的离群、版本连续性 56%、位姿差五栋里两栋超出、没有省 CPU），`Params.map_continue` 保持默认关，`RECON_OUTPUT_VERSION` 不变。重试前可以考虑的改动：只在照片数超过 `Params.small_set_images`（30）时续建（小照片集保留三个种子的多样性，那里的续建版本 18/20 在 2 m、3° 内，亚琛那样的退化都出在小照片集）；接受续建的门槛不用 `suspicious()` 的 5%（它拒掉了 76 次里的 40 次）；以及完整模型的重试规则（设计里的开放问题 5）。

### 6.9 一张图切出的扇区视图（持久地图第 2a 步）

`scripts/replay_map_views.py`，离线重放，不跑 SfM：对每个保留了稀疏模型的完整模型作业（只留了最好的那个连通块），在它的照片里每个可建立面的扇区切一个视图：这个扇区的立面作业会取的照片（扇区加 `face_margin_deg`）的相机，以及其中至少两台相机看到的点（`cut_view`）。视图按自己的相机配准一次（"视图"），按整个连通块配准一次（"连通块"），再像立面模型一样判：`qa.assess`（立面）、轮廓检验和参考检验，和完整模型作业跑完时页面上显示的该扇区立面模型比。"混合"：视图过了质量门就用视图，否则用立面模型（2b 设计里没被连通块覆盖的扇区退回立面作业）。2b 的规则：发布的立面不少、显示给用户的看错立面相机不多、参考位置和旋转误差不大。

重放（worktree 钉在 0520acd，v1-after-fixes 核心五栋两次重复，CPU 4 分 54 秒）：勃兰登堡门、比萨、圣母教堂有完整模型作业（洗礼堂、亚琛没有），18 个作业共 236 个（作业、扇区）对，每次运行最后一个完整模型作业 40 对。

| 范围 | 方案 | 发布的立面 | 显示的看错立面相机 / 有参考的相机 | 参考位置中位数 | 参考旋转中位数 | 轮廓旋转 |
|---|---|---|---|---|---|---|
| 全部 236 对 | 立面模型 | 126 | 101 / 2131 | 1.41 m | 0.85° | 6.0° |
| | 视图 | 88 | 26 / 1927 | 1.16 m | 0.57° | 4.7° |
| | 连通块 | 76 | 30 / 1791 | 1.33 m | 0.57° | 2.5° |
| | 混合 | 150 | 106 / 2475 | 1.26 m | 0.66° | 6.0° |
| 最后一个作业 40 对 | 立面模型 | 28 | 14 / 531 | 1.79 m | 0.85° | 7.2° |
| | 视图 | 14 | 4 / 436 | 1.24 m | 0.69° | 7.2° |
| | 混合 | 28 | 16 / 550 | 1.28 m | 0.76° | 6.7° |

- 236 个视图里 148 个没过质量门：107 个相机少于 3 台（这个扇区的照片不在完整模型留下的最好连通块里），29 个注册比例不够（视图注册比例中位数 0.44），16 个被标记的相机太多，16 个点太少。
- 过了门的视图更准：显示的看错立面相机 26 对 101，参考位置和旋转误差更小。但单用视图发布的立面少（88 对 126），混合发布的立面多（150 对 126）却多显示 5 台（最后一个作业：2 台）看错立面的相机。
- **结论：现在不做 2b**（按规则两种方案都不过）。局限：每个完整模型作业只留了最好的连通块，所以大多数没过门的视图是"没有相机"；要看保留全部连通块的效果，得用保留它们的运行重放（2b 本身会保留）。

### 6.10 冻结候选 F 对基线 B0（外部评审采纳计划的最终评测）

F = 提交 5e44646 的全部默认值（特征缓存开、续建关、#73 吸附门开、风格化走结构流水线、锚点下限 `dev-core5-v1`），B0 = 819cb78。结果文件都在 `eval/results/` 里：`synthetic-v1/{b0,v1-merged-s7,f}.json`、`mdx-two-v1/{b0,v1-merged-s7,f}.json`、`commons-ten-v1/{v1-after-fixes-s7,b0-dev,b0-test,f-dev,f-test,f-uncapped}.json`；2×2×2 拆分（吸附门、结构、谁的 SfM 产物）的 32 个补算格子在各测试集的 `ablation-v5/` 子目录（不进排行榜，也不进 `compare` 的噪声来源）。完整的表、方法和解读在 [REVIEW_ADOPTION.md](REVIEW_ADOPTION.md)。

- `compare --criteria review-v1`：dev FAIL（主要指标 134.85 → 136.8 m，+1.95 没超过 9.9 m 的噪声门槛；比萨的 `cov_geo_wrong_m_5` 0.5 → 2.0，不用缓存时也出现过），synthetic-v1、mdx-two-v1 FAIL（主要指标不变，要求变好）；test INSUFFICIENT（有数据的规则全部通过，标定只有 4 个模型、三栋没有发布的模型）。硬守卫在两个划分上都是 0。
- 作业 CPU 秒：核心五栋 22,497 → 9,057（−59.7%，和同一前缀、不用缓存的对照比），留出五栋 14,859 → 4,127（−72%，B0 当时机器较忙），synthetic-v1 −32%，mdx-two-v1 −29%。
- 留出集的标定：去重后 8 个模型（圣家堂 2、匈牙利国会大厦 6），只有一栋有 3 个以上，四个分量都是证据不足；匈牙利国会大厦的水平位置 6 个模型里 4 个在 3σ 外。`Settings.calibrated_anchor_ids` 保持为空。
- 不设上限的勃兰登堡门、圣母教堂各 1 次（`f-uncapped`）：都没截断，第一个发布的模型在到达计划处理 98.9 s、6.0 s 后出现，完整模型都没发布。

### 6.11 D31 的验证（照片资格和立面按张数过门）

D31 改了两处产品规则：照片里的建筑必须是主体（离 OSM 轮廓 ≤ 50 m，或估算占画面 ≥ 1/3，拒绝码 `not_subject`；ARCHITECTURE §3.2），立面模型注册率不到 60% 时注册 ≥ 6 张并拟合上 GPS 也能过质量门（`registration_path` = `"count"`；ARCHITECTURE §3.8）。基准这边同时修了印度门的参考（第 2 版，第 7.3 节）。通过标准 `eval/criteria/d31-v1.json` 在全新运行之前写好。全新运行 2026-11-03 21:10 到 11-04 00:20 UTC 跑完，**结论是 FAIL**：留出五栋那一对有三条规则不过（两条墙面覆盖的守卫、一条标定），要判的四项核对都通过（本节末尾的"结果"）。下面先是事先写好的做法和预测，原样保留。

**A（基线）是 F**，B 是 D31 的候选：

- synthetic-v1、mdx-two-v1、commons-ten-v1 的 dev：提交的 `f.json`、`f.json`、`f-dev.json` 原样用（dev 五栋和这两个测试集都不受印度门参考修正的影响）。
- commons-ten-v1 的 test：F 的 `f-test.json` 要按参考第 2 版重新打分，A、B 才用同一份参考：

  ```bash
  python -I scripts/benchmark.py rescore --testset commons-ten-v1 \
      --from eval/results/commons-ten-v1/f-test.json --name f-test-ref2 --allow-recon-diff \
      --reference-cache <缓存目录：印度门是第 2 版，其余四栋是 B0 的缓存>
  ```

  【补算，2026-11-03，在 09f7621 上；那时重建输出版本还是 1，不用 `--allow-recon-diff`，版本 2 之后要加】这样补算五栋：圣家堂、匈牙利国会大厦、林肯纪念堂、悉尼歌剧院的指标和 `f-test.json` 逐个相同（只少了 D29 删掉的七个上色指标），输入指纹不变；印度门只有按参考打分的指标变（第 7.3 节：有参考的已定位照片 6 → 3、错 3 → 0）。印度门记录的 `poses.reference.fix.version` 是 2。

  这个文件提交在 `eval/results/commons-ten-v1/d31-baseline/f-test-ref2.json`（和当时写出的逐字节相同，09f7621 的干净工作树补算的）。放在子目录里，和 `ablation-v5/` 一样不进排行榜，也不进 `compare` 自动收集的噪声来源：它补算的是 F 同样那两次 test 运行，放进 `commons-ten-v1/` 的话，F 的 test 运行在噪声来源里会出现两次（印度门的位姿指标还是另一版参考的极差），排行榜上也会多出一行只跑了五栋的 F。

**B（候选）**：一个同时带两处改动的候选，从事先写明的提交跑，命令和 F 一样（每次只跑一个 COLMAP）：synthetic-v1、mdx-two-v1 各 `--repeat 2`；commons-ten-v1 的 dev 五栋 `--repeat 2 --max-requests brandenburg-gate=104 frauenkirche-dresden=31`，test 五栋 `--repeat 2`；`--time-cap 3600`，风格化开。不做拆分对照：照片资格是确定性的，每个模型又记着自己走的是哪条注册路径。`--reference-cache` 可以指向 B0 的缓存目录：印度门那个文件的键对不上，会从本地的参考模型按第 2 版重建（约 10 s，并覆盖那个文件）；其余建筑的键不变。

**判定**：每一对结果跑一次 `compare --criteria d31-v1`（synthetic-v1、mdx-two-v1、dev、test 四对），再按 `d31-v1.json` 的 `checks` 另算六项：

- `reference_version_2`：A 的印度门记录 `poses.reference.fix.version` 是 2；B 用的参考缓存里印度门文件的 `reference_version` 是 2。`inputs_sha256` 不含参考版本，`compare` 自己看不出两边用的不是同一版参考。
- `acceptance_equals_replay`：每栋、每次重复的 `summary.json` 漏斗（上传、接收、各拒绝原因的张数；合成集还有 `subject_unchecked` 警告的张数）都要等于文件里写的期望值。期望值来自 D31 照片资格那一步的离线重放：产品自己的 `ingest_upload` 按 F 的上传顺序、在 F 的请求上限内重放 F 的每一次上传；除了 `not_subject`，每个原因的张数都和 F 一样。照片校验不依赖 SfM，所以对不上就是程序错误，不是噪声。
- `pose_correct_pooled_rise`：dev 和 test 两个结果十栋合在一起，`photo_pose_correct`（重复的中位数）之和要超出噪声地变多；噪声取 A、B 各自这十栋运行间极差之和里大的那个（和 `compare` 给分划分汇总定噪声的办法一样）。去掉印度门也要成立（划分文件的污染说明里，印度门的对错单独报）。F（印度门按第 2 版）是 167.5（去掉印度门 164.5），极差之和 3。只看 dev 预计变化在噪声附近，所以这条按两个划分合起来判。
- `count_path_models`：B 里每个按张数过门的模型（结果 `details.models[].qa_registration_path`，每次重复的 `summary.json` 里模型的 `qa_checks.registration_path`）逐个列出：建筑、重复、扇区、请求、注册 / 输入张数、点数、是否发布和显示、已验证参考相机数、位姿对 / 错、偏航误差。合起来，这些模型没被标记的已注册相机里有已验证参考位姿的（参考第 2 版），模型自己给的位姿在 10 m、15° 以内（`POSE_RIGHT_M`、`POSE_RIGHT_DEG`）算对，对的比例要 ≥ 0.745：这是 F 已发布立面的标尺（探针 B：134 个模型 2,427 个位姿里对 1,809 个）。这样的位姿不到 100 个就是 INSUFFICIENT。偏航误差 ≥ 30° 的（`wrong_face_models_published` 的判法）要是 0 个；没有偏航（已验证参考相机太少）的列为"无法核对"。完整模型走了 `"count"` 是程序错误。
- `wrong_face_reference_v1`（只报告，可选）：从提交 0ebbcc1（参考修正之前）的工作树、用 B0 的参考缓存把 B 补算一遍；错面模型只许出现在印度门的东面，逐个列出。
- `review_v1_raw_count`（只报告）：同样的 A、B 跑 `compare --criteria review-v1`，把它按张数判的 `photo_pose_wrong`（合并和每一栋）列在错误率那条旁边。

整个 D31 的结论：规则或核对有 FAIL 就是 FAIL，否则有 INSUFFICIENT 就是 INSUFFICIENT，否则 PASS。F 在 test 上按 `review-v1` 是 INSUFFICIENT（标定只有 4 个模型、三栋没有结构的值），D31 碰到同样的情况照样报 INSUFFICIENT 并写明原因。

**预测**【探针 B 在 F 的运行上的模拟和离线重放，样本内；不是全新运行的结果】：每次运行接收的照片 911 → 859（按 F 的请求上限）；按张数新过门的立面模型 F 两次运行共 75 个，按参考第 2 版数位姿对 570 / 713 = 0.799（第 1 版 0.784），错面 0 个（第 1 版 4 个，全是印度门东面）；十栋合并的 `photo_pose_correct` 每次运行约 167.5 → 207.5；错误率逐栋没有超出抽样噪声的变差（最接近的是勃兰登堡门，14/70 → 15/71，区间端点 −0.123；印度门 0/3 → 4/19，−0.365）；CPU 时间约为 F 的 1.0 倍。

**结果**【全新运行，2026-11-03 21:10 – 11-04 00:20 UTC；样本内：阈值和印度门的修正是看了 F 在这十栋上的运行定的（第 2 节"D31 的污染"），test 五栋对 D31 只是复现测量，不是留出检验】

- B = 提交 09f7621，钉住的工作树，没有未提交的改动（代码指纹 53f6d0fdee2e）。命令见上面"B（候选）"一段，和 F 一样。四个命令都正常结束，每栋每次重复都跑完、没有截断；命令带着 `--resume`，但没有一次重复是续跑的；同一时间只跑一栋。请求数和 F 相同（勃兰登堡门 104、圣母教堂 31 是上限），`inputs_sha256` 每栋都和 F 相同。
- 参考缓存是 B0 的那份，印度门换成第 2 版的文件（sha256 `fb611996…`）。14 个缓存键在 09f7621 下都对得上，没有重建任何参考。
- 结果文件：`eval/results/synthetic-v1/d31.json`、`eval/results/mdx-two-v1/d31.json`、`eval/results/commons-ten-v1/d31-dev.json`、`eval/results/commons-ten-v1/d31-test.json`；test 的 A 是上面的 `d31-baseline/f-test-ref2.json`。

`compare --criteria d31-v1` 的结论（A = F，B = D31，两次重复的中位数）：

| 对 | 结论 | 不过和证据不足的规则 |
|---|---|---|
| synthetic-v1 | INSUFFICIENT | 结构的两条守卫没有值（合成集没有分件快照）；8 条标定规则只有 12 个模型（要 20）。没有 FAIL |
| mdx-two-v1 | INSUFFICIENT | 结构守卫在亚琛没有值；标定只有 2 个模型。没有 FAIL |
| commons-ten-v1 dev | PASS | 没有 |
| commons-ten-v1 test | **FAIL** | `cov_local_wrong_m_1p5`、`cov_geo_wrong_m_5`、`calib_xy_beyond3` 不过；结构守卫在悉尼歌剧院没有值（它没有模型），是 INSUFFICIENT |

前两对证据不足的原因和 F 自己比自己时一样。有 FAIL 就是 FAIL，所以整个 D31 的结论是 FAIL。主要的规则（A → B）：

| 规则 | synthetic-v1 | mdx-two-v1 | dev | test |
|---|---|---|---|---|
| 主要指标 `cov_local_supported_m_1p5`，不变差 | 102 → 102 | 36 → 36 | 136.8 → 199.6 m（门槛 10） | 34 → 52 m（门槛 9.9） |
| 六条硬守卫，每栋每次都是 0 | 0（结构两条没有值） | 0（结构两条在亚琛没有值） | 0 | 0（结构两条在悉尼没有值） |
| `cov_local_wrong_m_1p5`，合并和每栋不变差 | 0 → 0 | 2 → 2 | 78.8 → 32.5 | **3 → 38.9（噪声 8）；圣家堂 0.5 → 15、印度门 0 → 19.9（门槛都是 1）** |
| `cov_geo_wrong_m_5`，同上 | 0 → 0 | 14 → 14 | 144.7 → 118.8 | **37 → 69（噪声 8）；圣家堂 34.5 → 65** |
| `wrong_face_cameras_shown` | 0 → 0 | 0 → 0 | 2 → 3（门槛 6） | 0 → 0 |
| `gt_gross_shown` | 0 → 0 | 0 → 0 | 6.5 → 1.5 | 2.5 → 4.5（门槛 3） |
| `photo_pose_wrong` 的错误率（95% 区间靠近 0 的端点） | 0/44 → 0/44（−0.080） | 9.5/23 → 4.5/19（−0.417） | 82.5/227 → 82.5/229.5（−0.091；最近的一栋是比萨，14/73 → 17.5/76.5，−0.095） | 35/58 → 38.5/101.5（−0.370；最近的一栋是林肯纪念堂，15/18 → 17.5/22，−0.270） |
| 标定 8 条 | 证据不足（n = 12） | 证据不足（n = 2） | 都过（n = 59；`calib_xy_beyond3` 2/59 = 0.034） | 7 条过（n = 22）；**`calib_xy_beyond3` 1.5/22 = 0.068 > 0.05** |
| `cpu_time_s` B / A，≤ 1.25 | 0.842 | 0.858 | 0.802 | 0.790 |

另外五条代价规则（`jobs_cpu_s`、`peak_rss_jobs_mb`、`request_latency_p90_s`、`photo_time_to_pose_p50_s`、`disk_run_mb`）四对都在 1.25 倍以内，最接近的是 mdx-two-v1 的 `photo_time_to_pose_p50_s`，1.23 倍。

核对（`d31-v1.json` 的 `checks`，按文件里写的方法读运行产物；两个核对者各自算，复核时又从原始文件重算了一遍，数都一致）：

| 核对 | 角色 | 结果 | 数 |
|---|---|---|---|
| `reference_version_2` | 前提 | PASS | A：`f-test-ref2` 的印度门 `poses.reference.fix.version` 是 2（13 台未验证），别的建筑都没有修正。B：09f7621 的 `REFERENCE_FIXES` 只有印度门第 2 版；缓存里印度门文件的 `reference_version` 是 2（41 台里 28 台已验证），和 B0 的第 1 版文件只差这 13 个标记，点和参考位姿都相同；另外 13 个缓存文件和 B0 的逐字节相同；两次重复都记着第 2 版、读的都是这份缓存。B 的结果 JSON 把它记在 `details.style.reference.fix`（每次重复的 `summary.json` 在 `poses.reference.fix`），补算的 A 记在 `poses.reference.fix` |
| `acceptance_equals_replay` | 守卫 | PASS | 14 栋 × 2 次 = 28 个漏斗（上传、接收、每个拒绝原因的张数）都等于文件里的期望值，结果里的 `photos_accepted` 也一样。`not_subject`：勃兰登堡门 6、比萨 3、匈牙利国会大厦 5、林肯纪念堂 7、印度门 25、悉尼歌剧院 15、mdx-two-v1 的亚琛 2（这 2 张 F 本来就拒了）；合成集只多了 `subject_unchecked` 警告，37 张和 12 张。十栋每次运行接收 911 → 859 张 |
| `pose_correct_pooled_rise` | 守卫 | PASS | 十栋合并的 `photo_pose_correct` 167.5 → 210.0（+42.5；噪声 12 = A、B 各自极差之和 3 和 12 里大的那个）；去掉印度门 164.5 → 194.0（+29.5） |
| `count_path_models` | 守卫 | PASS | 按张数过门并发布的模型 59 个，全是立面，没有完整模型。每次运行：亚琛 1 / 2（r1 / r2）、勃兰登堡门 8、比萨 2、圣家堂 14、林肯纪念堂 2、印度门 2（都是西面）。它们自己的位姿对 513 / 609 = 0.842（要 ≥ 0.745、≥ 100 个）；偏航 ≥ 30° 的 0 个，最大 22.2°（勃兰登堡门东南第 97 个请求，5 个有参考的位姿全错）；3 个没有偏航、无法核对（亚琛东面第 29 个请求两次、东南第 29 个请求一次，各只有 1–2 台已验证的参考相机）。另有 8 个（每次 4 个）按张数过了注册这一项，又因为点数不到 300 没过门：勃兰登堡门西北第 103 个（7/14，221 点）、印度门西面第 26 个（7/13，299 点）、圣家堂西面第 54 个（7/13，151–152 点）和第 60 个（9/16，220 点）；连它们一起算是 531 / 651 = 0.816，也过 |
| `wrong_face_reference_v1` | 只报告 | 符合要求 | 用 0ebbcc1 的工作树、B0 的缓存（印度门第 1 版）补算 B：只有印度门东面第 52 个请求的模型（16/25，2,227 点，按比例过门，偏航 −178.1°）算错面，每次重复 1 个，别的建筑都是 0。按第 1 版，印度门还会读成错面相机 3、错误位姿 6/22、`cov_local_wrong_m_1p5` 22.9 m |
| `review_v1_raw_count` | 只报告 | 报告 | 按张数数错误位姿，test 上圣家堂 5 → 9、林肯纪念堂 15 → 17.5、印度门 0 → 1 被标出（预计的就是这三栋），匈牙利国会大厦 15 → 11；合并 35 → 38.5，不显著。dev 82.5 → 82.5（勃兰登堡门 14 → 9.5 显著变好，比萨 14 → 17.5 在噪声内）。`review-v1` 自己的结论四对都是 FAIL：前三对没做到它要求的"变好"（主要指标、`jobs_cpu_s`），test 是上面同样的三行 |

**为什么不过**：

- `cov_local_wrong_m_1p5`、`cov_geo_wrong_m_5`（test）。B 的两次重复数一样（圣家堂 15 / 15、65 / 65，印度门 19.9 / 19.9），按参考第 1 版打分也一样，所以不是噪声，也不是参考版本的问题。
  - 圣家堂：最后显示的三个立面都是按张数过门的。南面第 65 个请求（15/28，3,551 点，偏航 −8.7°）覆盖 19 m 轮廓，只有 5 m 在 1.5 m 内：局部错 14 m，地理 5 m 错 19 m。西南第 59 个（18/40，偏航 15.3°）覆盖 15 m，局部对 12 m、错 3 m，地理错 13 m。这两个重建 F 的运行里就有（同样的作业，同样的注册 / 输入张数），F 的质量门按注册率 0.54、0.45 把它们拦下了，D31 的张数路径放了出来。圣家堂没有一张照片被资格规则去掉，所以这完全是张数路径的结果。它们的位姿不坏（圣家堂按张数过门的模型，相机对 294 / 334 = 0.88），错的是墙的位置。东北第 57 个（19/32，也走张数路径）的覆盖和 F 显示的东北第 53 个一模一样：局部支持 35 m，地理 5 m 错 35 m，这 35 m 在 F 里就有。三个立面的地理错墙按轮廓段取并集是 65 m（逐个加起来 67 m）。
  - 印度门：西面第 52 个请求（18/30，按比例过门，第一次显示）覆盖 19.9 m 轮廓，在地理坐标系里全在 5 m 内（地理错 0），但两个坐标系里都没有一段在 1.5 m 内。它放得对（偏航 1.5°，相机中位误差 3.2 m，7 个有参考的位姿全对），墙离 OSM 轮廓 1.5–5 m（残差中位数 1.78 m，90 分位 4.5 m）。F 在这里没有模型，守卫拿 19.9 m 和 0（噪声 0）比：一栋楼从"没有模型"变成"墙在 5 m 内、不在 1.5 m 内"的模型，按这条规则总是变差。它的注册率到 0.600，是因为资格规则去掉了它的 10 张输入（F 里同一个作业是 22/40 = 0.55）；没有资格规则，张数路径也会发布 F 那个 22/40 的模型（拟合上了 GPS，点数够），所以资格规则决定的是它走哪条路，不是西面有没有模型。
- `calib_xy_beyond3`（test）：1.5/22 = 0.068，刚过 0.05 的上限（r1 是 1/22 = 0.045，能过；r2 是 2/22；规则看中位数）。3σ 外的全是匈牙利国会大厦西面按比例过门的模型，D31 一个都没动：第 32 个请求（13/21）r1 z = 3.84、r2 3.40；第 41 个（29/38）只在 r2 是 6.02（F 里同一个作业、同样 38 张输入注册了 31–32 张，z = 1.99 / 1.54，是 SfM 的运行间差异）。F 也一样（r1 3 个模型里 1 个、r2 5 个里 3 个在 3σ 外），只是 F 只有 4 个模型（中位数），按 `min_n` = 20 是证据不足。B 多了 18 个按张数过门的模型（圣家堂 14、印度门 2、林肯纪念堂 2）和 2 个按比例过门的（圣家堂东北第 53 个、印度门西面第 52 个），n 到了 22，这条规则才判得出来；这 20 个模型水平方向的 |z| 都不超过 1.6（最大 1.602；偏航、尺度、倾斜的 z 另算，`calib_xy_beyond3` 只看水平）。所以这是匈牙利国会大厦的锚点水平 σ 本来就说小了（6.10 节），数据多了才判出来，不是 D31 造成的。
- 这三条都不在上面的预测里：探针 B 的终态打分只算了位姿、错面和 `gt_gross`，没算 `cov_*` 和标定。圣家堂那两个模型在 F 的产物里就有，本来可以事先打分。

**分建筑**（commons-ten-v1，两次重复的中位数，A → B）。"张数"是按张数过门并发布的模型数；位姿一列是有参考位姿、已定位的照片里对 / 错 / 合计（参考第 2 版）；覆盖三列的单位是米：

| 建筑 | 划分 | 接收（`not_subject`） | 发布（张数） | 最后显示的面 | SfM 定位的照片 | 位姿 对 / 错 / 合计 | 局部 1.5 m 支持 | 局部错 | 地理 5 m 错 | CPU B / A |
|---|---|---|---|---|---|---|---|---|---|---|
| 洗礼堂 | dev | 51 → 51（0） | 6 → 6（0） | E NE SE SW W，不变 | 38 → 38 | 14/5/19 → 14/5/19 | 35 → 35 | 5 → 5 | 16 → 16.5 | 0.90 |
| 亚琛 | dev | 75 → 75（0） | 2 → 3.5（r1 1、r2 2） | N → N E，r2 还有 SE | 13 → 21 | 0/0/0 → 1.5/0/1.5 | 0 → 0 | 0 → 0 | 0 → 0 | 0.90 |
| 勃兰登堡门 | dev | 141 → 136（6） | 22 → 28（8） | E NE NW SE W → E NE SE W | 85 → 78 | 56/14/70 → 56.5/9.5/66 | 13.9 → 72.7 | 47.3 → 1 | 28.4 → 1 | 0.74 |
| 比萨斜塔 | dev | 118 → 115（3） | 11 → 11（2） | E SW W → r1 E NE SW W 加完整模型，r2 E SW W | 75 → 78.5 | 59/14/73 → 59/17.5/76.5 | 8 → 12 | 8 → 7 | 2 → 2 | 0.74 |
| 圣母教堂 | dev | 83 → 83（0） | 20 → 20（0） | N NE S SW W，不变 | 63 → 64.5 | 15.5/49.5/65 → 16/50.5/66.5 | 79.9 → 79.9 | 18.5 → 19.5 | 98.3 → 99.3 | 0.85 |
| 圣家堂 | test | 128 → 128（0） | 1 → 15（14） | NE → NE S SW，都走张数路径 | 15 → 47 | 15/5/20 → 38/9/47 | 34 → 52 | 0.5 → 15 | 34.5 → 65 | 0.81 |
| 匈牙利国会大厦 | test | 91 → 86（5） | 5 → 4.5（0） | E SE W，不变 | 42 → 40.5 | 2/15/17 → 4.5/11/15.5 | 0 → 0 | 2.5 → 2.5 | 2.5 → 2.5 | 0.87 |
| 林肯纪念堂 | test | 62 → 58（7） | 0 → 2（2） | 无 → E（张数路径） | 0 → 10 | 3/15/18 → 4.5/17.5/22 | 0 → 0 | 0 → 1.5 | 0 → 1.5 | 0.61 |
| 印度门 | test | 86 → 65（25） | 0 → 4（2，后来被替换） | 无 → E、W（都按比例） | 0 → 44 | 3/0/3 → 16/1/17 | 0 → 0 | 0 → 19.9 | 0 → 0 | 0.55 |
| 悉尼歌剧院 | test | 76 → 62（15） | 0 → 0 | 无 → 无 | 0 → 0 | 没有参考位姿 | 0 → 0 | 0 → 0 | 0 → 0 | 0.88 |

- 勃兰登堡门不再显示西北面：F 显示的西北立面有 6 张注册照片，其中一张离轮廓 76 m，被资格规则去掉；剩下 5 张、约 112 点，过不了门。后来按张数过了注册的西北第 103 个请求（7/14）只有 221 点，也没过。F 那个西北模型 6 台有参考的相机全是粗差，支持 0 m 墙。
- 林肯纪念堂的两个模型是东面第 39、44 个请求（10/29、10/32），显示的是第 44 个。它们自己的相机只对 19 / 39，是按张数过门的模型里最弱的（勃兰登堡门 0.82、比萨 0.88、圣家堂 0.88）。印度门东面第 52 个请求（16/25）的 16 张照片没法核对：那一侧的参考相机全在第 2 版标成未验证的 13 台里。悉尼歌剧院两次都没有一个作业建出模型。
- mdx-two-v1 的亚琛：F 两次都发布了 1 个东南立面（4 个有参考的位姿全错），B 两次都没有：同样 6 张照片的东南作业在 SfM 这一步找不到初始图像对。这一步在质量门之前，接收的照片也和 F 一样，不是这两条规则直接造成的。
- CPU 时间省的 20% 左右有一部分是机器：B 和 F 的库版本相同，内核构建不同（F 是 6.18.44-fc-v185，B 是 fc-v230）；接收、作业和模型都和 F 一样的洗礼堂、圣母教堂也是 0.90、0.85 倍。悉尼歌剧院不能当对照（去掉了 14 张、少了 4 个作业）。不管怎么算，代价规则都过。

**被资格规则去掉的照片**（逐张追到 F 自己的运行里）：十栋每次运行有 61 张被标 `not_subject`，其中 52 张是 F 接收过的（911 → 859），另外 9 张 F 本来就因为别的原因拒了。

- 勃兰登堡门：5 张里 4 张在 F 发布的立面里注册上了。F 显示的模型上有参考位姿的 3 张，r1 全错（偏 22.4 m、13.9 m、10.5 m），r2 错 2 张、对 1 张（"Komuniści przed Bramą Brandenburską"，5.1 m）；第 4 张在 F 里被标记，只按 GPS 显示；离轮廓 76 m 的那张是上面说的西北立面。整栋没有变差：对的位姿 56 → 56.5，错的 14 → 9.5。
- 比萨：3 张都注册在 F 发布的西南、西面立面里；2 张有参考位姿，但 F 在显示的模型里把它们标记了（GPS 偏了 84 m 和 146 m），只按 GPS 显示；第 3 张没有参考。对的位姿 59 → 59。
- 印度门：21 张里 6 张注册过，都在 F 被拒的西面模型里，从没显示过，F 里没有一张被判对或错。去掉它们，西面第 52 个请求的注册率才到 60%。
- 林肯纪念堂：4 张里 1 张注册过，在 F 被拒的东南模型里，它的 GPS 偏了 459 m。悉尼歌剧院：14 张里 2 张注册过，也在 F 被拒的东南模型里。匈牙利国会大厦：5 张都没注册过。
- 合起来：两次运行里这条规则只丢了一个对的位姿（勃兰登堡门 r2），每次运行去掉 2–3 个错的位姿和一个坏的立面；没有一栋对的位姿变少。

**预测和结果**：

| 预测【探针 B 和离线重放，样本内】 | 预测 | 全新运行 |
|---|---|---|
| 每次运行接收的照片 | 911 → 859 | 911 → 859，每栋每个原因都一样 |
| 按张数过门并发布的立面模型（两次合计） | 75 | 59 |
| 它们的位姿对的比例（参考第 2 版） | 570 / 713 = 0.799 | 513 / 609 = 0.842 |
| 错面模型，参考第 2 版 | 0 | 0 |
| 错面模型，参考第 1 版 | 4 个，都是印度门东面按张数过门的 | 每次 1 个：印度门东面第 52 个请求，按比例过门 |
| 十栋合并的 `photo_pose_correct` | 167.5 → 207.5（去掉印度门 192.5） | 167.5 → 210.0（194.0） |
| 错误率逐栋 | 没有一栋超出抽样噪声地变差 | 成立；最近的是比萨（dev，−0.095）和林肯纪念堂（test，−0.270） |
| CPU 时间 | 约 F 的 1.0 倍 | 0.79–0.86 倍（一部分是机器，见上） |

按张数过门的模型比预测少，只少在资格规则去掉了照片、作业顺序因此变了的楼上：勃兰登堡门每次 8 个（预测 11）、林肯纪念堂 2 个（3）、印度门 2 个（6：预测的东面、西面张数模型在 B 里成了按比例过门的东面、西面第 52 个）。探针 B 重放的是 F 的照片集合。亚琛、比萨、圣家堂和预测逐个相同。

**复现**：

```bash
python -I scripts/benchmark.py compare eval/results/synthetic-v1/f.json eval/results/synthetic-v1/d31.json --criteria d31-v1
python -I scripts/benchmark.py compare eval/results/mdx-two-v1/f.json eval/results/mdx-two-v1/d31.json --criteria d31-v1
python -I scripts/benchmark.py compare eval/results/commons-ten-v1/f-dev.json \
    eval/results/commons-ten-v1/d31-dev.json --criteria d31-v1
python -I scripts/benchmark.py compare eval/results/commons-ten-v1/d31-baseline/f-test-ref2.json \
    eval/results/commons-ten-v1/d31-test.json --criteria d31-v1
```

上面的数是这些结果进仓库之前跑的 `compare`：噪声来源是 A、B 和 09f7621 的 `eval/results`。现在 `d31-dev`、`d31-test` 也在 `commons-ten-v1/` 里，成了噪声来源（第 8 节），再跑时 dev 的两条覆盖守卫门槛 6 → 8 m、5 → 8 m，test 主要指标的门槛 9.9 → 10 m；每条规则的结论都不变（synthetic-v1、mdx-two-v1 的输出完全相同）。要逐字节复现，加 `--results-dir` 指向 09f7621 的 `eval/results`（例如 `git archive 09f7621 eval/results | tar -x -C <目录>`，再用 `<目录>/eval/results`）。六项核对要读运行目录（`summary.json`、数据目录、参考缓存），它们在仓库外，没有提交。

照片资格的两处取法是负责人还没答复的默认值（50 m 是负责人给的数，量的是相机 GPS 到 OSM 主轮廓）：占画面的门槛取 1/3，另一种选法是 1/2；没有绑定 OSM 轮廓的地标现在不查、照片带警告，另一种选法是先绑定轮廓才收照片。

### 6.12 D30 的验证（视觉定位：同一批产物上的配对补算）

D30 的视觉定位（`crowd3d/localize.py`：把没有显示中的 SfM 位姿、同意公开位置、带 EXIF 35 mm 焦距的照片，用固定的 EXIF 焦距做 PnP，定位到已保存的地图里）默认关着（`Settings.localize_photos`）。第十一步按 `eval/criteria/d30-v1.json` 决定要不要打开它。这份标准在 B 跑之前写好（D30 设计验证计划的第 4 条），门槛是设计里定的，写在任何定位器出现之前。下面先是事先写好的做法和预测，原样保留；结果在本节末尾：**`localize_photos` 保持关闭**，代价规则在 commons-ten-v1 的两个划分上都不过，准确和隐私方面的规则和核对全部通过。

**A 和 B**：都是 D31 全新运行的补算，从同一个干净的提交跑，A 不加、B 加 `--localize`（6.4 节）。两个结果打分的是同一批 SfM 产物、同一份代码，差别只能来自定位器，所以比较时用 `--noise-factor 0`：运行间的极差不进门槛，只剩每个指标自己的最小变化量。第二步在 d5e51fa 上算过的 A 只在暂存目录里，没有 `photo_pose_loc_wrong`，也早于第三到第九步，要在 B 的提交上重跑一遍（几分钟）。

```bash
# 四对，每对 A、B 各一条；参考缓存是 D31 全新运行用的那份（印度门第 2 版）
python -I scripts/benchmark.py rescore --testset commons-ten-v1 --from eval/results/commons-ten-v1/d31-dev.json \
    --name d30-a-dev --reference-cache <暂存>/d31/fresh/refs/commons-ten-v1
python -I scripts/benchmark.py rescore --testset commons-ten-v1 --from eval/results/commons-ten-v1/d31-dev.json \
    --name d30-b-dev --reference-cache <暂存>/d31/fresh/refs/commons-ten-v1 --localize
# test：d31-test.json → d30-a-test / d30-b-test；synthetic-v1、mdx-two-v1：d31.json → d30-a / d30-b
python -I scripts/benchmark.py compare eval/results/commons-ten-v1/d30-a-dev.json \
    eval/results/commons-ten-v1/d30-b-dev.json --criteria d30-v1 --noise-factor 0
```

每次只跑一栋、一个 COLMAP 进程，`nice 19`，2 个线程（`localize.LOCALIZE_THREADS`）。D31 的运行是重建输出版本 2 的产物，不用 `--allow-recon-diff`。

**打分的就是产品会显示的**：`--localize` 原样调用产品的 `candidate_maps`（这次运行 `data/maps` 下保存的地图）、`eligible`、`load_sparse_maps` 和 `ColmapLocalizer.localize`（配对是 `select_pairs`：候选地图里每张不同的照片，上限 200），记录按候选地图的 `maps_key` 存，和定位作业一样；吸附用打分时用的那些，同意按打分时的假设（这些库早于同意功能，每张通过的上传都算同意）；被接受的定位按 `localize_photos` 打开时产品的规则显示（`jobs.shown_localizations`）。基准在运行结束时给每张照片打分，`/poses` 只列最近一次重建作业结束前上传的照片：这个差别由核对 `fresh_run_display_equals_score` 管。

**CPU 怎么算**（代价规则就看它）：这些运行没留下特征缓存，特征要重新提。活的系统里地图照片的特征已经在地标的缓存里（重建作业提过），所以每次重复先把地图照片的特征放进这次重复自己的缓存（从这栋楼的特征库里拿，库里没有才提；这一步的 CPU 单独记在 `poses.runs[].loc.map_features`，不算指标），再在这个缓存上跑定位器：查询照片的特征、每个图像对的匹配和 PnP 都由定位器自己算，这一次调用的进程 CPU 就是 `photo_pose_loc_cpu_s`。重复之间不共用查询特征和图像对，所以 r2 和 r1 是一样量的（设计里原来打算让 r2 用 r1 的图像对省时间，那样 r2 的数会偏小，两次的中位数也跟着偏小）。这个数**既不是活系统的上界、也不是下界**：活系统的缓存里还有重建作业匹配过的图像对（commons-ten-v1 的 D31 运行里，查询照片和地图照片在同一个穷举匹配的重建作业里出现过的图像对占 10–56%），定位器就不用再算；可活系统在候选地图每变一次时都会把照片再试一遍（新地图的照片要新匹配）。核对 `loc_cost_live` 用一次全新运行量出活系统的数，放在旁边。

**规则**（每个划分，B 对 A）：

| 规则 | 要求 | 为什么 |
|---|---|---|
| `photo_pose_loc_wrong` | 守卫，`zero`：每栋、每次重复都是 0；没有参考位姿的建筑留出（`skip_without` `photo_pose_gt`） | 硬门槛：不许有一张定位出的位姿显示错（别人看到的那一行离参考位姿超过 10 m 或 15°）。取每栋各次重复的最大值，所以等于两个划分合起来是 0。为这条门槛在第 8 版里原地加了这个指标（还没有提交过第 8 版的结果）。悉尼歌剧院没有参考位姿，`pose_metrics` 给它的这个指标是空的：在行里点名留出，不然 test 上永远是 INSUFFICIENT（在 B 跑之前改的）；有参考位姿却没有值的建筑仍是 INSUFFICIENT |
| `photo_pose_loc_wrong_face` | 守卫，`zero`，留出同上 | 硬门槛：看错面的定位位姿 |
| `photo_pose_wrong_share` | 守卫，`not_worse_per_building` | 合并和每一栋都不比 A 差，门槛 0.01。有了上面的零门槛，定位只会让一栋楼的错误比例变小，所以这条只会和 `photo_pose_loc_wrong` 一起不过 |
| `photo_pose_loc_cpu_s` | 代价，`max_share` 0.25 of `cpu_time_s` | 定位的 CPU 不超过运行自己 CPU 的四分之一（每个划分，各栋取重复的中位数再加起来；`cpu_time_s` 是源运行的，原样拷来） |
| `photo_pose_loc_uncertainty_coverage` | 只报告 | 附 B 合并的个数和 95% Wilson 区间；两个划分合起来的数在核对 `loc_coverage_pooled` 里，不到 10 张是 INSUFFICIENT（照实报，不算通过） |
| `photo_pose_correct_rate_testset` | 只报告 | 定位带来的提高（配对、`--noise-factor 0`）。评估设计里"dev 上要显著变好"不作规则：预计只多几张，会挡下一个无害的功能，和"不能浪费"矛盾 |

另外只报告：定位出的张数、其中有参考位姿的和对的、试过的张数、每张的 CPU、严格对的比例、全部照片的确定比例和覆盖、EXIF 张数和扣下的位姿。

**核对**（`compare` 不判，按文件里写的方法读运行产物和 `loc.json`）：

- `paired_rescores`（前提）：两边的源结果 sha256、提交和代码指纹相同、工作树干净，参考缓存相同，只有 B 的 `provenance` 有 `localize`，`compare` 报告配置相同、没有输入不同的建筑。
- `only_loc_changes`（守卫）：每栋每次重复，除了 `loc` 以外的每个来源，B 的张数加上 B 从这个来源挪去定位的张数（`loc.replaced`）等于 A 的张数；`photo_pose_*` 以外的指标全都相同。别处有差别就是程序错误。
- `loc_gps_conflict`（守卫，属于硬门槛）：两个划分每栋每次重复的 `loc.gps_conflict` 加起来，错的要是 0，数都报出来。定位位姿离自己的 GPS 再远也照样显示（设计的粗糙值第 5 条：记下、不拦），前提就是它们里面没有一张错；有一张错，就要像被标记的 SfM 相机那样拦下（`Params.loc_withhold_gps_conflict`）。
- `loc_coverage_pooled`（报告）：B 两个划分所有重复合起来的 `photo_pose_loc_within_uncertainty / photo_pose_loc_gt` 和 Wilson 区间。
- `synthetic_truth`（前提）：synthetic-v1 的 B 里每张定位出的照片离生成器的真值 1 m 内，去掉它所在地图自己的朝向误差（这张地图里相机朝向误差的中位数：生成器的罗盘有 8° 噪声，配准出的地图本身就偏几度，定位器继承它）后朝向差 1° 内；`photo_pose_loc_correct` 等于 `photo_pose_loc_gt`。
- `fresh_run_display_equals_score`（守卫）：一次全新运行（洗礼堂，`--repeat 1`，配置里 `localize_photos` 打开）：在它数据目录的副本上起 API，`/poses` 各来源的张数等于 `as_of` 之前上传的照片的 `photo_pose_sfm`、`_loc`、`_exif`，每一行等于内部位姿的公开行，泄漏扫描通过；对同一次运行做 `rescore --localize`，得到的定位记录和这次运行库里的相同（状态、模型、位姿差 1 cm 内）；`photo_pose_not_uploaded` 不比 D31 的洗礼堂运行多（定位作业不许吃掉时间上限）。
- `privacy_tests`（前提）：B 的提交上快速测试全过（尤其 `tests/test_api.py::test_no_public_get_serves_per_photo_data`），仓库里没有 `loc.json`、逐张记录或 Commons 运行的库内 id。
- `loc_cost_live`（报告）：那次全新运行里定位作业的 CPU（作业日志），放在同一栋楼补算的 `photo_pose_loc_cpu_s` 旁边。
- `loc_report`（报告）：`loc` 块的记录、原因、类别、EXIF → 定位的前后对错、每张地图的表、扣下的位姿。

第十一步：规则全过、角色是守卫和前提的核对全过，才打开 `localize_photos`；有一项 FAIL 或 INSUFFICIENT 就保持关闭，并写明是哪一项、为什么。

**预测**【不是 B 的结果：产品的 `candidate_maps`、`eligible`、`select_pairs` 在 D31 运行副本上的计数（纯函数，不跑 COLMAP；同意按补算的假设），加上第七、八步的冒烟运行（产品定位器跑在 D31 运行的副本上：synthetic-two-sides r1、印度门 test r1、比萨 dev r1、洗礼堂 dev r1、匈牙利国会大厦 test r1）。这几次冒烟是写这份标准之前看过的，门槛没有因此改】

| 建筑 | 划分 | 每次重复要试的照片（r1 / r2） | 图像对（r1 / r2） | 冒烟 |
|---|---|---|---|---|
| 洗礼堂 | dev | 11 / 11 | 396 / 396 | r1 定位出 2 张，有参考的 1 张对（4.72 m、1.42°） |
| 亚琛 | dev | 34 / 31 | 646 / 713 | |
| 勃兰登堡门 | dev | 21 / 21 | 1,680 / 1,680 | |
| 比萨斜塔 | dev | 17 / 29 | 1,530 / 2,117 | r1 定位出 1 张（没有参考）；因为内点率被拒的一张按参考其实是对的 |
| 圣母教堂 | dev | 0 / 0 | 0 | 没定位上的照片要么相机被标记、要么没有焦距 |
| 圣家堂 | test | 54 / 54 | 2,808 / 2,808 | |
| 匈牙利国会大厦 | test | 52 / 34 | 832 / 1,530 | r1 一张也没定位出 |
| 林肯纪念堂 | test | 44 / 44 | 440 / 440 | |
| 印度门 | test | 13 / 13 | 442 / 442 | r1 定位出 1 张（没有参考） |
| 悉尼歌剧院 | test | 0 / 0 | 0 | 没有地图 |
| synthetic-two-sides | synthetic-v1 | 1 / 1 | 11 / 11 | r1 定位出（2,071 / 2,076 个内点） |
| mdx-two-v1 两栋 | | 0 / 0 | 0 | 照片没有 EXIF 焦距 |

- 没有一组候选地图碰到 200 对的上限（最多 90 张地图照片）。
- `photo_pose_loc_wrong`：有参考位姿的建筑上都是 0（目前见过的唯一一张有参考的定位位姿是对的）；test 那一行写成“0 on 4 building(s); left out without photo_pose_gt: sydney-opera-house”。`photo_pose_loc_wrong_face` 一样。有参考的定位位姿预计两个划分加起来不到 10 张，所以 `photo_pose_loc_uncertainty_coverage` 是 INSUFFICIENT，零门槛也只有很少的证据。
- `photo_pose_correct_rate_testset`：每个划分每次重复多 0–2 张，dev 上低于最小变化量 0.01（每次重复约 230 张有参考的照片）。
- 代价：按冒烟的代价（印度门 r1 47 张图、442 对共 187 CPU-s，比萨 r1 107 张图、1,530 对共 777 CPU-s：每对约 0.2–0.4 CPU-s，每提一张图约 2 CPU-s），dev 约 1,100–2,000 CPU-s，预算是 0.25 × 7,410 = 1,852，过不过都有可能；test 约 1,300–2,300 CPU-s，预算是 0.25 × 3,356 = 839，**预计不过**（圣家堂一栋 54 张照片 × 52 张地图照片就约 600–1,200 CPU-s）；synthetic-v1 几个 CPU-s，预算 24；mdx-two-v1 是 0。如果实测和估计差不多，代价规则在 test 上不过，第十一步就不会打开 `localize_photos`。

**结果**【补算，2026-11-04 12:25–14:45 UTC；A、B 都从 716de16 的干净工作树跑（代码指纹 `9c148b35f001`），指标第 8 版；产物和参考缓存都是 D31 全新运行的（印度门第 2 版）；同一时间只跑一个 COLMAP 进程】

- 结果文件：A 是 `eval/results/synthetic-v1/d30-a.json`、`mdx-two-v1/d30-a.json`、`commons-ten-v1/d30-a-dev.json`、`commons-ten-v1/d30-a-test.json`；B 在各测试集的 `d30-localize/` 子目录里（`d30-b.json`、`d30-b-dev.json`、`d30-b-test.json`），下面两次全新运行和对它的补算也在 `commons-ten-v1/d30-localize/`。都是当时写出的文件原样拷进来的。
  - B 和 A 打分的是同样两次 SfM 运行，而且 B 是定位打开时的样子、不是产品的默认，所以放进子目录：不进排行榜，也不进 `compare` 自动收集的噪声来源。
  - A 留在测试集目录里：它是现在的产品（定位关着）在第 8 版下的分数，也是第 8 版的第一份噪声记录（D31 的 `d31*` 是第 7 版，不算第 8 版的噪声来源，第 8 节）。
- 墙钟：B 的 dev 36.5 min、test 39.4 min、synthetic-v1 8 s、mdx-two-v1 2 s；A 几分钟。B 每次重复写一份 `loc.json`（库内照片 id 和精确位姿），只在暂存目录里。

**结论：`localize_photos` 保持关闭，第十一步不改默认值。** 代价规则在 commons-ten-v1 的两个划分上都不过（按写下的规则算；dev 那一条落在测出的机器速度漂移之内，不稳，只有 test 那一条是稳的，见表下的"代价的分子和分母不是同一台机器测的"）；核对 `fresh_run_display_equals_score` 按写下的做法也不过（`run` 还不给运行里的定位位姿打分，见下面"全新运行"）。准确和隐私方面的规则和核对全部通过。跑评测的、两个核对者和最后的复核各自算了一遍，结论和每个门槛上的数都一致。

| 对 | `compare --criteria d30-v1 --noise-factor 0` | 不过的规则 |
|---|---|---|
| synthetic-v1 | PASS | 没有（定位 CPU 2.45 / 96.4 = 0.025） |
| mdx-two-v1 | PASS | 没有（照片没有 EXIF 焦距，一张要试的都没有） |
| commons-ten-v1 dev | **FAIL**（不稳） | `photo_pose_loc_cpu_s`：1,978.95 / 7,409.6 = 0.267 > 0.25（预算 1,852.4 CPU-s，超了 126.6，即 7%；按机器速度漂移换算约 0.17–0.19，会过） |
| commons-ten-v1 test | **FAIL** | `photo_pose_loc_cpu_s`：2,231.95 / 3,355.6 = 0.665 > 0.25（预算 838.9 CPU-s；不算圣家堂也是 0.53；按机器速度漂移换算约 0.41–0.46，仍然不过） |

规则（A → B，两次重复的中位数）：

| 规则 | synthetic-v1 | mdx-two-v1 | dev | test |
|---|---|---|---|---|
| `photo_pose_loc_wrong`（零） | 0（2 栋） | 0（2 栋） | 0（5 栋） | 0（4 栋；悉尼歌剧院没有参考位姿，留出） |
| `photo_pose_loc_wrong_face`（零） | 0 | 0 | 0 | 0 |
| `photo_pose_wrong_share`（合并和每栋都不变差） | 0 → 0 | 0.3529 → 0.3529 | 0.3552 → 0.3423（两次合计 157/442 → 153/447）；洗礼堂 0.3529 → 0.2941、勃兰登堡门 0.1439 → 0.1250、比萨 0.2186 → 0.2176，没有一栋变差 | 0.3575 → 0.3575 |
| `photo_pose_loc_cpu_s` ≤ 0.25 × `cpu_time_s` | 0.025 | 0 | **0.267** | **0.665** |
| 只报告：`photo_pose_correct_rate_testset` | 0.88 → 0.88 | 0.2245 → 0.2245 | 0.4703 → 0.4851（每次重复对的 142.5 → 147） | 0.2938 → 0.2938 |
| 只报告：`photo_pose_loc_uncertainty_coverage` | 1/1 | 没有定位位姿 | 5.5/5.5（Wilson 0.589–1.000） | 没有有参考的定位位姿 |

**代价的分子和分母不是同一台机器测的**（第一轮复核发现，结论不变）：

- 代价规则的分母 `cpu_time_s` 是从 D31 全新运行原样拷来的（2026-11-03 21:13 起，内核构建 6.18.44-fc-v230）；分子 `photo_pose_loc_cpu_s` 是 B 在 2026-11-04 12:44–14:00 测的（容器重启以后，fc-v233）。
- 同样的工作，这两段时间用的 CPU 差了约 1.5 倍。洗礼堂定位关着的对照（`d30-fresh-control`，14:04 起；定位关着时 D30 加的作业检查马上返回；和 D31 一样是 51 张通过的照片、8 个作业、6 个发布的模型）对 D31 这栋的两次：
  - `cpu_time_s` 422.5 对 285.7 / 280.1，`jobs_cpu_s` 411.7 对 278.5 / 272.8，都是 1.48–1.51 倍；
  - 单线程 Python 的 `struct_time_s` 9.9 对 6.9 / 6.4（1.43–1.55 倍），`style_time_s` 11.6 对 7.9 / 7.5（1.47–1.55 倍）：每一步都一起慢，看来是机器慢了，不是哪一步多做了事；
  - 定位打开的那次全新运行（12:37 起）去掉定位作业的 284.0 CPU-s，重建作业还用了 437.4 CPU-s（1.57–1.60 倍）。B 正好在这两次运行之间跑。
- 所以分子是在慢了约 1.5 倍的机器上测的，分母不是。按 1.43–1.60 倍换算：dev 的 1,978.95 CPU-s 约合 1,235–1,379，是运行的 0.17–0.19，会过；test 的 2,231.95 约合 1,392–1,555，是 0.41–0.46，仍然不过。dev 只要两边差 7% 判定就会翻过来，test 要差 2.66 倍才会。
- 漂移只在洗礼堂一栋上测过，换算是估计，不是每个划分上的测量，所以表里照规则写的仍是 FAIL。稳的只有 test 那一条。结论不变：test 的代价和 `fresh_run_display_equals_score` 都不过，`localize_photos` 保持关闭。
- 下一次跑 B，代价的分子和分母要在同一台机器、同一段时间里测：要么每个划分再跑一次定位关的全新运行当分母；要么同时跑一次对照（例如洗礼堂定位关的全新运行），和源结果里同一栋楼比出倍数，按倍数换算。对照的 CPU 和倍数要写在代价规则的结果旁边。

核对（按文件里写的方法读运行产物和 `loc.json`）：

| 核对 | 角色 | 结果 | 数 |
|---|---|---|---|
| `paired_rescores` | 前提 | PASS | 四对都是：源结果 sha256、提交（716de16）、代码指纹相同，工作树干净，第 8 版，只有 B 有 `provenance.localize`，参考相同（印度门第 2 版），生效参数相同，没有输入不同的建筑 |
| `only_loc_changes` | 守卫 | PASS | 2,288 条逐张记录：2,269 条相同，19 条是定位出的张次（commons-ten-v1 17、synthetic-v1 2），别的变化 0；`photo_pose_*` 以外的指标没有一个不同；每次重复按来源的账都对得上 |
| `loc_gps_conflict` | 守卫 | PASS（没有被检验） | 显示的 19 个定位位姿里没有一个被 GPS 核对标出（Commons 上离自己 GPS 1.93–26.03 m），所以错的是 0 只因为这一组是空的 |
| `loc_coverage_pooled` | 报告 | 11/11 | Commons 两个划分所有重复合计 11/11 张次（Wilson 0.741–1.000），只是 6 张不同的照片（0.61–1.00）；加上合成集 13/13 |
| `synthetic_truth` | 前提 | PASS | 合成两面的那一张两次都定位出：离真值 0.531 m，去掉地图自己 3.27° 的朝向误差后差 −0.111°；`photo_pose_loc_correct` = `photo_pose_loc_gt` = 1 |
| `fresh_run_display_equals_score` | 守卫 | **FAIL（按写下的做法）** | 产品这边一致，`run` 自己的打分没算定位位姿，见下面"全新运行" |
| `privacy_tests` | 前提 | PASS | 716de16 上快速测试 1,053 个全过，node 119 个通过、1 个跳过；2,352 个库内照片 id（A、B 的逐张记录、每份 `loc.json`、两次全新运行的库）在当时 266 个跟踪的文件里一次都没出现，17 个接受的 Commons 精确位姿也没有；`loc.json`、逐张记录和留一法的文件都没进仓库。这次提交的 11 个结果文件又扫了一遍：没有库内 id、精确位姿或上传照片的键（只有印度门参考里 13 台未验证相机的 Commons 文件名，`d31-test.json` 里本来就有） |
| `loc_cost_live` | 报告 | 0.39 | 见下面"全新运行" |
| `loc_report` | 报告 | | 下面两张表 |

**每栋**（commons-ten-v1 和合成两面；要试的照片和图像对是每次重复的数（r1 / r2），和上面的预测逐个相同；定位 CPU 是两次的中位数；地图特征是这栋楼两次重复合计、重提地图照片特征的 CPU，不算指标）：

| 建筑 | 划分 | 要试的照片 | 图像对 | 接受并显示 | 有参考：对 / 错 | 定位 CPU-s | 地图特征 CPU-s |
|---|---|---|---|---|---|---|---|
| 洗礼堂 | dev | 11 / 11 | 396 / 396 | 2 / 2 | 每次 1 / 0 | 252.8 | 124.4 |
| 亚琛 | dev | 34 / 31 | 646 / 713 | 0 / 0 | | 444.25 | 89.7 |
| 勃兰登堡门 | dev | 21 / 21 | 1,680 / 1,680 | 4 / 4 | 每次 4 / 0 | 567.7 | 200.5 |
| 比萨斜塔 | dev | 17 / 29 | 1,530 / 2,117 | 1 / 2 | r2 1 / 0 | 714.2 | 249.7 |
| 圣母教堂 | dev | 0 / 0 | 0 | 0 | | 0 | 0 |
| 圣家堂 | test | 54 / 54 | 2,808 / 2,808 | 0 / 0 | | 1,292.1 | 155.2 |
| 匈牙利国会大厦 | test | 52 / 34 | 832 / 1,530 | 0 / 0 | | 592.7 | 115.9 |
| 林肯纪念堂 | test | 44 / 44 | 440 / 440 | 0 / 0 | | 225.45 | 23.3 |
| 印度门 | test | 13 / 13 | 442 / 442 | 1 / 1 | 没有参考 | 121.7 | 71.7 |
| 悉尼歌剧院 | test | 0 / 0 | 0 | 0 | | 0 | 0（没有地图） |
| synthetic-two-sides | synthetic-v1 | 1 / 1 | 11 / 11 | 1 / 1 | 每次 1 / 0 | 2.45 | 6.0 |

- Commons 两次重复合计试了 483 张次（dev 175、test 308）、18,900 个图像对。接受 17 张次（dev 15、test 2），都显示了。
- 被拒 466 张次（dev 160、test 306）：配上的对应太少（`too_few_matches`）134 / 290，自由焦距核对对不上（`focal_mismatch`）7 / 11，内点率 8 / 3，内点数 6 / 2，重投影 2 / 0，焦距不合理 3 / 0。
- 没有试、记成跳过的 413 张次（dev 191、test 222）：没有 EXIF 焦距 166 / 136，相机在显示的模型里被标记 25 / 20，没有地图 0 / 66。
- 圣母教堂有 5 张候选地图，但要试的照片都没有焦距或相机被标记。圣家堂两次各试 54 张、一张也没定位上，每次花约 1,300 CPU-s，一栋就超了 test 的预算。

**定位出了什么**（11 个有参考的张次，是 6 张不同的照片，全对；打分的是别人看到的那一行）：

- 原来显示 EXIF、是错的 → 对：4 张次（勃兰登堡门、洗礼堂各一张，两次都是）。
- 原来显示 EXIF、是对的 → 仍然对：2 张次（勃兰登堡门一张）。
- 原来只有 GPS、不显示 → 对：5 张次（勃兰登堡门两张，各两次；比萨一张，只在 r2）。
- 没有参考、核对不了的 6 张次：洗礼堂一张原来只有 GPS，比萨、印度门各一张原来是 EXIF，都是两次。
- 离参考 0.11–7.11 m、0.77–4.56°；5 m / 10° 内的 8 / 11。公开的范围都是 `sp` 16 m、`sy` 21°（洗礼堂没参考的那张是 23 m / 25°）：定位的 σ 是地图在那一点的 SfM σ 再加相对误差（`hypot(sfm, 1.5 m)`），所以虽然对，画出来的范围很宽。
- dev 每次重复对的位姿 142.5 → 147、严格对的 89 → 92，`photo_pose_correct_rate_testset` 0.4703 → 0.4851；test 不变（印度门那张没有参考）。预测是"每个划分每次重复多 0–2 张、有参考的不到 10 张"：实际 dev 多 4.5 张、有参考的 11 张次，比预测多；要试的张数、图像对和代价和预测一样。

**留一法标定**（只在暂存目录、不进仓库：文件里有库内 id 和精确位姿。n 比 B 大得多，但这些照片 SfM 本来就能注册，比真的查询容易，只能当标定看，不能当结论）：r1，八栋有地图的楼，160 张有参考、SfM 定位过的照片，每张先从所有含它的地图里拿掉（pycolmap `deregister_frame`），再用产品的 `ColmapLocalizer` 定位，共 4,269 CPU-s。

- 接受 110 张：对 84、错 26（24%，95% Wilson 0.167–0.324）。26 张错的里，13 张自己的 SfM 位姿也是错的（地图传下来的错）；另外 13 张自己的 SfM 位姿是对的：SfM 对的 89 张里 13 张（14.6%）定位后变错（有的在圣家堂只比 15° 多不到 0.5°，有的离自己的 SfM 相机 12.6 m）。定位器自己也会放错，B 的"11 张次 0 错"证据很薄：0/11 时错误率的 95% 上界约 0.26，按 6 张照片算约 0.39。
- 落在自己的 2σ 范围里 105 / 110（Wilson 0.898–0.980）；错面 0；GPS 冲突 0。
- 和自己的 SfM 相机比：中位 1.16 m / 0.13°，90 分位 5.14 m / 0.62°，最大 12.6 m / 5.0°。
- 被拒 50 张：自由焦距核对对不上 39（林肯纪念堂 9 张里 8 张），重投影 5，焦距不合理 3，内点率 2，内点数 1。这条核对可能太严，没细查。

**全新运行**（核对 `fresh_run_display_equals_score` 和 `loc_cost_live`）：洗礼堂，`--repeat 1`，716de16，配置 `eval/configs/d30-localize.json`（默认值加上 `localize_photos` 打开；基准每次上传都同意公开位置），结果 `commons-ten-v1/d30-localize/d30-fresh-loc.json`，墙钟 378.4 s；同一提交、默认配置（定位关）的对照 `d30-fresh-control.json`；对它做的 `rescore --localize` 是 `d30-fresh-loc-rescore-b.json`。

- **产品这边一致**。在数据目录的副本上起 API（不启动作业），读所有不用登录的 GET：
  - `/poses` 的 `as_of` 等于 `poses_snapshot_time`，51 张通过的照片都在快照里；`counts`：照片 51、同意 51、显示 40（sfm 35、loc 2、exif 3）、没确定 11。
  - 这等于对快照里的照片用 `jobs.shown_localizations` 按基准规则打分的结果，也等于 `rescore --localize` 的结果（sfm 35、loc 2、exif 3；有参考的那张对）。
  - 40 行都等于内部位姿的 `public_row`，也等于按基准规则（加上库里显示的定位结果）打出的那一行的公开字段，只有白名单里的键。
  - 泄漏扫描（`test_api` 的规则用在真实数据上）0 个问题：没有库内 id、密钥、撤回凭证、服务器路径、逐张照片的键，没有 0.1 m 精度的相机 ENU、GPS、上传时间；5 个点云、6 个风格化 GLB 里没有库内 id；20 个定位作业的 `/api/jobs/{id}` 都返回 404。有 3 处"和照片距离差不到 0.05 m"是整数计数碰巧接近（例如 19.951 m 和"20 台相机"），不是泄漏，但测试里的严格规则用在真实库上会报出来。
  - 定位记录：库里 29 条；补算在最终的候选地图键下得到 16 条，和库里同一个键下的 16 条逐条相同（状态、原因、模型、位姿差 0.0000 m）。另外 13 条在两个旧键下，都是现在按 SfM 显示的照片：`eligible` 不会再试它们，sfm 又排在 loc 前面，所以不显示，但也没有东西删掉它们（其中 1 条是接受的）。
  - `photo_pose_not_uploaded` 是 0，和 D31 一样：定位作业没有吃掉时间上限。
- **按写下的做法不过**：这次运行自己记下的指标是 sfm 35 / loc 0 / exif 4、`photo_pose_loc_queries` 0，`/poses` 显示的是 sfm 35 / loc 2 / exif 3。原因是 `run_crowd` 在运行结束打分时没有把库里的定位结果交给打分（要接 `jobs.shown_localizations`，第九步说留给第十步，没有接上）。产品、基准规则和补算三者一致，差的是 `run` 自己的打分。
- `run` 的汇总还把定位作业算进了作业指标：`jobs_run` 8 → 28，`registration_rate` 0.9464 → 0.9298（定位作业的照片进了分母）。基准的作业在上传请求里同步跑，定位作业也一样：请求延迟中位数 41.5 → 55 s，`photo_time_to_pose_p50_s` 118 → 177 s。
- **活系统的代价**（`loc_cost_live`）：20 个定位作业，试了 21 次、接受 4 次（定位表每张照片一行，候选地图变了会覆盖，所以库里是 3 条接受的：2 条现在的、1 条旧键下的）；墙钟 141.5 s，作业日志合计 284.0 CPU-s，占这次运行 733.5 CPU-s 的 0.39。同一次运行的补算试了 11 张、236.0 CPU-s。20 个作业里 14 个什么也没试，只写了 `no_maps` / `no_focal` 的跳过记录：上传请求里的判断只看有没有在当前候选地图下还没有记录的照片，没有焦距、没有地图的照片也算（要记成跳过），所以候选地图每变一次又要建一个作业重新记一遍。
- **打开定位后，重建这边的指标也变了**（各只有一次，分不清是不是运行间的差别）：轮廓转角 1.6 → 10.21°（对照和 D31 的两次都是 1.6°），罗盘误差中位数 2.815 → 4.28°，地理尺度误差 10.995 → 10.725%，地理相机误差 7.415 → 7.375 m。可能的机制：定位作业把它算出的特征和图像对存进地标的缓存，之后的重建作业用上了（最后两个立面作业的缓存命中 25 → 38、146 → 179 对）。打开定位之前要重复几次全新运行的 A/B（开 / 关）；如果确实如此，打开定位就改变了重建输出，要按 `RECON_OUTPUT_VERSION` 的规则处理（6.4 节）。
- 对照的 CPU 时间 422.5 s 比 D31 这栋的两次（285.7、280.1 s）多一半，墙钟 214.5 s 对 145.3 s，同时没有别的任务。单线程的步骤也一样慢，看来是机器慢了（D31 在内核构建 fc-v230 上跑，D30 的运行在容器重启后的 fc-v233 上）。代价规则的分母是从 D31 原样拷来的 `cpu_time_s`，所以 dev 的代价判定不稳，见上面"代价的分子和分母不是同一台机器测的"。

**A 对 D31：第 8 版的规则改了什么**（A 和每次 D31 运行自己的逐张记录逐张比，只在暂存目录里，因为有照片的键；A 和第二步在 d5e51fa 上算的 A 逐条相同，2,288 条）：138 个张次不同，没有一个解释不了。

- 69 个：磁北或没标注的罗盘不再显示（负责人问题 1 的默认答复）。其中 23 个有参考：原来对的 4 个、错的 19 个。
- 37 个：坐标系没定准（只有先验、没吸附）的模型里的相机不再显示。commons-ten-v1 洗礼堂每次 3 张，mdx-two-v1 洗礼堂每次 3 张，匈牙利国会大厦 r1 有一个这样的模型（25 张）；其中匈牙利国会大厦 r1 有 2 张原来对的位姿，换成了一个旧的、定准了的模型里错的相机。设计里预计的"mdx 亚琛 4 张"没有出现：它唯一的模型在 D31 里没有发布。
- 32 个：取整到 1 m / 1° 以后，判定跨过了 5 m / 10° 或 10 m / 15° 的线，两个方向都有。
- `photo_pose_*` 以外的指标 0 个不同。每次重复：dev 确定的照片 337 → 321、328 → 313，对的 145 → 140、149 → 145；test 确定的 206 → 173、207 → 186，对的 62 → 60、64 → 64。

**现在的产品显示什么**（A，定位关着；两次重复的中位数；补算假定每张通过的照片都同意公开）：

| | dev（核心五栋） | test（留出五栋） |
|---|---|---|
| 通过的照片 | 460 | 399 |
| 显示位姿的（SfM / 手机 GPS + 真北罗盘） | 317（277 / 40） | 179.5（134 / 45.5） |
| 只有 GPS、不显示 | 143 | 219.5 |
| 显示的里有参考的：对 / 错（10 m / 15°） | 142.5 / 78.5 | 62 / 34.5 |
| SfM 位姿对的比例 | 137 / 210.5 = 0.65 | 61 / 82.5 = 0.74 |
| EXIF 位姿对的比例 | 5.5 / 10.5 | 1 / 14 |
| 落在自己 2σ 范围里（全部 / SfM / EXIF） | 0.860 / 0.860 / 0.857 | 0.876 / 0.885 / 0.821 |
| 扣下的位姿：本来对的 / 本来错的 | 10.5 / 17.5 | 5 / 14.5 |

显示出来的位姿约三分之一离参考超过 10 m 或 15°，但 86–88% 落在它自己标出的 2σ 范围里：范围是诚实的，只是宽。EXIF 那一行在留出集上 14 张有参考的只有 1 张在 10 m / 15° 内，82% 在范围里，所以标"未经模型核对"、在地上画范围圆。

**复现**：

```bash
python -I scripts/benchmark.py compare eval/results/commons-ten-v1/d30-a-dev.json \
    eval/results/commons-ten-v1/d30-localize/d30-b-dev.json --criteria d30-v1 --noise-factor 0
# test：d30-a-test.json 对 d30-localize/d30-b-test.json；synthetic-v1、mdx-two-v1：d30-a.json 对 d30-localize/d30-b.json
```

结果进仓库以后 `d30-a-dev`、`d30-a-test` 都在 `commons-ten-v1/` 里，都是第 8 版的噪声来源，dev 的比较会混进 test 的汇总极差（第 8 节），所以 `compare` 表里的噪声列和进仓库前不同；`--noise-factor 0` 下每条规则的值和结论都不变（进仓库后重跑过，`Criteria d30-v1` 一节逐字相同）。核对要读运行目录、`loc.json` 和数据库，它们在仓库外，没有提交。

**为什么不打开，打开要什么**：

1. 代价。dev 只超了 126.6 CPU-s，在机器速度漂移之内，换算后会过；test 不算圣家堂也是 0.53，换算后整体约 0.41–0.46，是稳的不过。圣家堂一栋每次 1,292 CPU-s、一张也没定位上（54 张查询 × 52 张地图照片，几乎全是 `too_few_matches`）。要过这条，定位器要便宜得多：现在每张查询和候选地图里每张照片都匹配一遍（`select_pairs`），设计里的检索（VISUAL_LOCALIZATION §3.3）还没做；也可以先判断哪些查询没希望。
2. `run` 还不给运行里的定位位姿打分：`run_crowd` 要接上 `jobs.shown_localizations`，作业指标（`jobs_run`、`registration_rate`）要把定位作业排除在外。
3. 打开定位以后重建这边的指标在一次运行里变了（轮廓转角 1.6 → 10.21°），要重复的全新 A/B 才分得清。
4. 证据少：有参考的只有 6 张不同的照片；留一法里接受的定位 24% 是错的，一半是定位器自己的错，"0 错"这条门槛多一些数据就可能不过。
5. 重跑 B 时，代价的分子和分母在同一台机器、同一段时间里测（上面"代价的分子和分母不是同一台机器测的"最后一条）。

改了其中任何一条都要重新跑 B 再判。如果不改定位器就打开（例如按活系统的代价算、去掉一些楼或者放宽门槛），那是看过 B 之后的新决定，要由负责人做，并照实写明。

### 6.13 D34 的验证（地理配准怎么权衡 GPS 和罗盘：离线重放）

评审重建部分有三条发现说地理配准（`reconstruct.georegister`）权衡证据的方式不对。评审没有在真实模型上量过，所以 D34 把三条改法各做成一个开关，先都按原来的行为放着，用 6.4 节的 `rescore --replay` 在 D31 全新运行保留的稀疏模型上重放，不跑新的 SfM。下面先是事先写好的做法和规则，原样保留；结果在后面：**(a) 和 (b)（没标真北的读数不算）一起打开，在 dev 上合格，是合格的组里开关最少的，现在是默认值。(c) 加在它们上面，够不上规则里"有用"的任何一条，保持默认关。留出集上跑的是三个一起开的那一组，通过了；(c) 在留出集上一次也没有裁决过，所以那次通过就是 (a) + (b) 的结果。单独打开哪一个都不合格。**

这一节在评审之后更正过一次（2026-11-12）：原来写的是"只有三个一起开的这一组合格，三个开关都成为默认值"。当时提交的七组里没有只开两个开关的；补跑以后，(a) + (b) 自己就合格。改了哪些话，列在本节末尾的"评审之后的更正"。

| 开关（`Params`） | 打开以后 | 对应的发现 |
|---|---|---|
| (a) `gps_span_model_side` | "GPS 能不能定尺度"的跨度和 GPS 的尺度、偏航 σ，在拟合出的相机布局上量（内点相机的 SfM 坐标乘以拟合尺度），不在带噪声的 GPS 坐标上量 | 站在同一个点拍的照片有 15–44% 被当成已配准，尺度差 2–4 倍 |
| (b) `compass_other` = `weak` 或 `ignored` | 没有标成真北的罗盘读数（磁北或没写参照）按 45° 的 σ 算，或者不算。偏航先验、相机的罗盘标记、候选排序、质量门的罗盘检查都按它 | 罗盘不分真北、磁北一律当真 |
| (c) `yaw_arbitration` | 偏航冲突（GPS 偏航和罗盘偏航差超过 40°）按强弱裁决：GPS 拟合强（内点 ≥ 8，冲突 ≥ 5 倍它自己的偏航 σ）而罗盘先验弱（折合不到 3 个真北读数，或者彼此相差超过 2 倍自己的 σ）时保留 GPS 拟合，这些读数不再标记相机、也不进质量门的罗盘检查；别的冲突照旧归罗盘 | 一个罗盘读数就能否掉 20 个 GPS 定出来的模型 |

第四个开关 `Settings.facing_away_warns_other_refs`（没标真北的背对朝向只警告、不拒绝）改变收下哪些照片，重放不出来，这里不量。

**怎么跑**。八条命令，从同一个干净的提交跑，都不跑 COLMAP：

```bash
# 不重放的补算（重放是否忠实，拿它对）
python -I scripts/benchmark.py rescore --testset commons-ten-v1 --from eval/results/commons-ten-v1/d31-dev.json \
    --name d34-plain-dev --restylize --reference-cache <暂存>/d31/fresh/refs/commons-ten-v1
# 七组重放：off（对照，三个开关明写成老规则）、a、b-weak、b-ignored、c、abc-weak、abc-ignored
python -I scripts/benchmark.py rescore --testset commons-ten-v1 --from eval/results/commons-ten-v1/d31-dev.json \
    --name d34-<组>-dev --restylize --replay eval/configs/d34-regeo/<组>.json \
    --reference-cache <暂存>/d31/fresh/refs/commons-ten-v1
python -I scripts/benchmark.py georeg <八个结果> --split dev --json <文件>
```

评审之后又用同样的命令、从同一个提交补跑了三组只开两个开关的：`ab-ignored`（(a) + (b) `ignored`）、`bc-ignored`（(b) `ignored` + (c)）、`ac`（(a) + (c)）。它们的配置当时不在那个提交里，`--replay` 给的是仓库外的文件，内容和现在提交的 `eval/configs/d34-regeo/<组>.json` 逐字节相同（结果里记着 sha256，有测试核对）。

这些命令是在重建输出版本 2 的代码上跑的；默认值改了以后（版本 3、4）重跑，每条要加 `--allow-recon-diff`（6.4 节）。`abc-weak`、`abc-ignored` 是三个开关一起开，(b) 各取一种。重放改的是每个模型的地理配准、相机标记和质量门结论；SfM 的结果（注册了哪些照片、点、建图时选了哪个候选）不变。它看不到的三件事见 6.4 节。

**事先写下的规则**（dev）。一组配置要同时满足"有用"和"没坏处"，它打开的开关才可以成为默认值。比的是它和 `off` 那一组；误差是 `georeg` 表里能和参考比的模型的（同一个模型在两次重复里只算一次），计数是划分的汇总（每栋两次重复的中位数再加起来）。

- 有用（满足一条就行）：
  - 尺度偏差超过 35% 的模型少 2 个以上，或者 |ln 尺度| 的均方根低 5% 以上；
  - 偏航误差的均方根低 5% 以上，或者偏航差超过 15° 的模型少 2 个以上；
  - 显示出来的错位姿（`photo_pose_wrong`）少 2 个以上，对的不减少；
  - 对的位姿（`photo_pose_correct`）多 2 个以上，错的不增加。
- 没坏处（都要满足）：
  - 发布的模型最多少 1 个；显示的位姿最多少 1%；对的位姿最多少 1 个；
  - 超出自己 2σ 范围的位姿最多多 1 个；
  - 看错面的模型和相机（`wrong_face_models_published`、`wrong_face_cameras_shown`）不增加；
  - 偏航、尺度、水平位置、倾斜四个 2σ 覆盖率各自最多降 0.02；
  - 尺度偏差超过 35% 的模型、偏航差超过 15° 的模型各自最多多 1 个。
- 几组都合格时选开关最少的那组，除非开关多的一组在"有用"的某一条上又好出同样的幅度。
- 一组都不合格时，三个开关都保持关，留出集上不跑任何打开开关的配置。

这条规则 2026-11-11 写在这一步的工作笔记里，早于任何一组的完整结果。要说明的是：写之前已经在没提交的代码上做过一遍探索，看过每个模型的尺度和偏航误差；写之后、这个提交之前，那一遍探索的各组完整结果也看过了，并且据此改了一处代码（被裁决否掉的罗盘读数不再进相机标记和质量门；之前裁决保住的模型会被质量门用同一个读数否掉）。下面的结果是从这个提交重跑的。

**dev 的结果**。十一条命令都从 65781fd 的干净工作树跑（原来的八条 2026-11-11，补跑的三条 2026-11-12；十个重放结果的 `code_sha256` 相同），结果文件原样放在 `eval/results/commons-ten-v1/d34-regeo/`（`d34-<组>-dev.json`；放在子目录里，不进排行榜，也不算噪声来源）。下面的表由 `georeg --split dev` 算出；`ac`、`ab-ignored`、`bc-ignored` 三行是补跑的，另外七行和补跑之前逐字相同。

重放先要忠实，才谈得上比较：

| 比什么 | 结果 |
|---|---|
| dev 两次重复共 238 个模型，重放了几个 | 238 个，没有原样留下的；用运行自己的参数重放出的相机，和存下的最大差 0.0000 m |
| 对照组 `off` 改了什么 | 变换变了 0 个，配准等级变了 0 个，发布状态变了 0 个。有 6 个模型的消息文字不同，都是亚琛只注册了 3 张的模型："3 个 GPS 里 2 个一致、跨度 X m、定不了尺度"那句话里的 X 不一样，因为三台相机时 RANSAC 的几对打平，保留的模型列图像的顺序又和作业当时不同。这 6 个模型的相机位置、标记、等级和质量门结论都和存下的一样 |
| `off` 对 `plain`：每栋、每次重复的每个指标（1590 个值） | 都相同；另有 27 个值不同，全是风格化那一步的用时和内存，同一份代码跑两遍也不一样 |
| `off` 对 `plain`：标定表 118 行 | 都相同 |
| `off` 对已提交的 `d30-a-dev.json`（716de16 上对同一批运行的补算）：1590 个值 | 除了 28 个用时和内存的值（`struct_peak_rss_mb`、`struct_time_s`、`style_time_s`）都相同；标定表 118 行都相同 |

**总体**

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 68.5 | 83 | 0.353 | 16/83 | 9.59 | 4.91 | 13/83 | 3.57 | 317 | 142.5 | 78.5 | 31 | 0.9661 / 0.9322 / 0.9322 / 0.9831 |
| a | 68.5 | 82 | 0.354 | 16/82 | 9.29 | 4.68 | 13/82 | 3.6 | 317 | 143.5 | 77.5 | 29 | 0.9661 / 0.9322 / 0.9322 / 0.9831 |
| b-weak | 71.5 | 85 | 0.349 | 16/85 | 11.09 | 4.57 | 13/85 | 3.57 | 317 | 144 | 77 | 35 | 0.8852 / 0.918 / 0.918 / 0.9836 |
| b-ignored | 68.5 | 79 | 0.361 | 16/79 | 9.99 | 4.21 | 12/79 | 3.33 | 317 | 145.5 | 75.5 | 33 | 0.9737 / 0.9298 / 0.9298 / 0.9825 |
| c | 68.5 | 83 | 0.353 | 16/83 | 9.59 | 4.91 | 13/83 | 3.57 | 317 | 142.5 | 78.5 | 31 | 0.9661 / 0.9322 / 0.9322 / 0.9831 |
| ac | 68.5 | 82 | 0.354 | 16/82 | 9.29 | 4.68 | 13/82 | 3.6 | 317 | 143.5 | 77.5 | 29 | 0.9661 / 0.9322 / 0.9322 / 0.9831 |
| ab-ignored | 68.5 | 79 | 0.362 | 16/79 | 9.69 | 4.22 | 11/79 | 3.33 | 317 | 149 | 72 | 31 | 0.9737 / 0.9298 / 0.9298 / 0.9825 |
| bc-ignored | 68.5 | 81 | 0.359 | 17/81 | 9.89 | 4.21 | 12/81 | 3.33 | 317 | 145.5 | 75.5 | 32 | 0.9746 / 0.9322 / 0.9153 / 0.9831 |
| abc-weak | 71.5 | 85 | 0.35 | 16/85 | 10.91 | 4.43 | 13/85 | 3.56 | 317 | 145 | 76 | 32.5 | 0.9016 / 0.9344 / 0.918 / 0.9836 |
| abc-ignored | 68.5 | 81 | 0.36 | 17/81 | 9.59 | 4.22 | 11/81 | 3.33 | 317 | 149 | 72 | 30 | 0.9746 / 0.9322 / 0.9153 / 0.9831 |

**配准等级，和重放改了什么**（两次重复都算）

| 组 | 模型 | 重放了 | 原样留下 | 变换变了 | 等级变了 | 发布状态变了 | full / prior_only / none | 已发布 |
|---|---|---|---|---|---|---|---|---|
| off | 238 | 238 | 0 | 0 | 0 | 0 | 189 / 49 / 0 | 137 |
| a | 238 | 238 | 0 | 189 | 2 | 0 | 187 / 51 / 0 | 137 |
| b-weak | 238 | 238 | 0 | 142 | 2 | 6 | 187 / 51 / 0 | 143 |
| b-ignored | 238 | 238 | 0 | 162 | 9 | 12 | 180 / 58 / 0 | 137 |
| c | 238 | 238 | 0 | 0 | 0 | 0 | 189 / 49 / 0 | 137 |
| ac | 238 | 238 | 0 | 189 | 2 | 0 | 187 / 51 / 0 | 137 |
| ab-ignored | 238 | 238 | 0 | 215 | 9 | 12 | 180 / 58 / 0 | 137 |
| bc-ignored | 238 | 238 | 0 | 162 | 3 | 12 | 186 / 52 / 0 | 137 |
| abc-weak | 238 | 238 | 0 | 195 | 4 | 6 | 185 / 53 / 0 | 143 |
| abc-ignored | 238 | 238 | 0 | 215 | 3 | 12 | 186 / 52 / 0 | 137 |

**按模型注册的照片数分档**（每格：模型数，|ln 尺度| 均方根，尺度偏差 > 35% 的个数，偏航均方根 °）

| 组 | 3-6 张 | 7-10 张 | 11-20 张 | 21+ 张 |
|---|---|---|---|---|
| off | 3，0.859，2/3，15.63 | 19，0.437，8/19，13.72 | 20，0.36，4/20，9.39 | 41，0.215，2/41，6.17 |
| a | 3，0.856，2/3，14.89 | 18，0.447，8/18，13.44 | 20，0.36，4/20，9.23 | 41，0.216，2/41，5.97 |
| b-weak | 3，0.892，2/3，25.34 | 19，0.431，8/19，16.18 | 22，0.343，4/22，9.15 | 41，0.214，2/41，6.47 |
| b-ignored | 3，0.875，2/3，18.8 | 19，0.435，8/19，13.91 | 22，0.343，4/22，9.17 | 35，0.229，2/35，6.1 |
| c | 3，0.859，2/3，15.63 | 19，0.437，8/19，13.72 | 20，0.36，4/20，9.39 | 41，0.215，2/41，6.17 |
| ac | 3，0.856，2/3，14.89 | 18，0.447，8/18，13.44 | 20，0.36，4/20，9.23 | 41，0.216，2/41，5.97 |
| ab-ignored | 3，0.869，2/3，17.65 | 19，0.437，8/19，13.55 | 22，0.344，4/22，9.03 | 35，0.23，2/35，5.84 |
| bc-ignored | 3，0.875，2/3，18.8 | 20，0.432，9/20，13.57 | 22，0.343，4/22，9.17 | 36，0.226，2/36，6.09 |
| abc-weak | 3，0.891，2/3，24.73 | 19，0.431，8/19，16.03 | 22，0.344，4/22，9.01 | 41，0.214，2/41，6.28 |
| abc-ignored | 3，0.869，2/3，17.65 | 20，0.434，9/20，13.23 | 22，0.344，4/22，9.03 | 36，0.226，2/36，5.83 |

**佛罗伦萨洗礼堂**（`florence-baptistery`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 6 | 3 | 0.131 | 0/3 | 5.63 | 0.61 | 0/3 | 4.61 | 39 | 11 | 6 | 0 | 1 / 1 / 1 / 1 |
| a | 6 | 3 | 0.129 | 0/3 | 5.25 | 0.77 | 0/3 | 4.61 | 39 | 12 | 5 | 0 | 1 / 1 / 1 / 1 |
| b-weak | 6 | 3 | 0.131 | 0/3 | 5.61 | 0.61 | 0/3 | 4.61 | 39 | 11 | 6 | 0 | 1 / 1 / 1 / 1 |
| b-ignored | 6 | 3 | 0.131 | 0/3 | 5.61 | 0.61 | 0/3 | 4.61 | 39 | 11 | 6 | 0 | 1 / 1 / 1 / 1 |
| c | 6 | 3 | 0.131 | 0/3 | 5.63 | 0.61 | 0/3 | 4.61 | 39 | 11 | 6 | 0 | 1 / 1 / 1 / 1 |
| ac | 6 | 3 | 0.129 | 0/3 | 5.25 | 0.77 | 0/3 | 4.61 | 39 | 12 | 5 | 0 | 1 / 1 / 1 / 1 |
| ab-ignored | 6 | 3 | 0.129 | 0/3 | 5.17 | 0.77 | 0/3 | 4.61 | 39 | 12 | 5 | 0 | 1 / 1 / 1 / 1 |
| bc-ignored | 6 | 3 | 0.131 | 0/3 | 5.61 | 0.61 | 0/3 | 4.61 | 39 | 11 | 6 | 0 | 1 / 1 / 1 / 1 |
| abc-weak | 6 | 3 | 0.129 | 0/3 | 5.18 | 0.77 | 0/3 | 4.61 | 39 | 12 | 5 | 0 | 1 / 1 / 1 / 1 |
| abc-ignored | 6 | 3 | 0.129 | 0/3 | 5.17 | 0.77 | 0/3 | 4.61 | 39 | 12 | 5 | 0 | 1 / 1 / 1 / 1 |

**亚琛大教堂**（`aachen-cathedral`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |
| a | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |
| b-weak | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |
| b-ignored | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |
| c | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |
| ac | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |
| ab-ignored | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |
| bc-ignored | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |
| abc-weak | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |
| abc-ignored | 3.5 | 0 | — | 0/0 | — | — | 0/0 | — | 41.5 | 1.5 | 0 | 0 | — / — / — / — |

**勃兰登堡门**（`brandenburg-gate`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 28 | 35 | 0.304 | 6/35 | 6.91 | 4.76 | 1/35 | 2.73 | 84 | 56.5 | 9.5 | 7 | 1 / 0.9643 / 0.9643 / 1 |
| a | 28 | 34 | 0.307 | 6/34 | 6.57 | 4.59 | 1/34 | 2.59 | 84 | 56.5 | 9.5 | 5 | 1 / 0.9643 / 0.9643 / 1 |
| b-weak | 28 | 34 | 0.306 | 6/34 | 7.21 | 4.1 | 1/34 | 2.59 | 84 | 56.5 | 9.5 | 7 | 0.9643 / 0.9643 / 0.9643 / 1 |
| b-ignored | 28 | 34 | 0.307 | 6/34 | 6.19 | 3.98 | 1/34 | 2.59 | 84 | 56.5 | 9.5 | 5 | 1 / 0.9643 / 0.9643 / 1 |
| c | 28 | 35 | 0.304 | 6/35 | 6.91 | 4.76 | 1/35 | 2.73 | 84 | 56.5 | 9.5 | 7 | 1 / 0.9643 / 0.9643 / 1 |
| ac | 28 | 34 | 0.307 | 6/34 | 6.57 | 4.59 | 1/34 | 2.59 | 84 | 56.5 | 9.5 | 5 | 1 / 0.9643 / 0.9643 / 1 |
| ab-ignored | 28 | 34 | 0.308 | 6/34 | 5.88 | 3.79 | 0/34 | 2.59 | 84 | 58.5 | 7.5 | 5 | 1 / 0.9643 / 0.9643 / 1 |
| bc-ignored | 28 | 34 | 0.307 | 6/34 | 6.19 | 3.98 | 1/34 | 2.59 | 84 | 56.5 | 9.5 | 5 | 1 / 0.9643 / 0.9643 / 1 |
| abc-weak | 28 | 34 | 0.306 | 6/34 | 7.04 | 3.91 | 1/34 | 2.59 | 84 | 56.5 | 9.5 | 5 | 1 / 0.9643 / 0.9643 / 1 |
| abc-ignored | 28 | 34 | 0.308 | 6/34 | 5.88 | 3.79 | 0/34 | 2.59 | 84 | 58.5 | 7.5 | 5 | 1 / 0.9643 / 0.9643 / 1 |

**比萨斜塔**（`leaning-tower-pisa`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 11 | 19 | 0.157 | 1/19 | 11.79 | 3.96 | 4/19 | 6.08 | 88 | 59 | 17 | 8.5 | 0.8636 / 1 / 1 / 0.9091 |
| a | 11 | 19 | 0.157 | 1/19 | 11.58 | 3.95 | 4/19 | 6.07 | 88 | 59 | 17 | 8.5 | 0.8636 / 1 / 1 / 0.9091 |
| b-weak | 12 | 21 | 0.15 | 1/21 | 11.35 | 4.21 | 4/21 | 3.64 | 88 | 59 | 17 | 9 | 0.875 / 1 / 1 / 0.9167 |
| b-ignored | 12 | 21 | 0.15 | 1/21 | 11.36 | 4.24 | 4/21 | 3.64 | 88 | 59 | 17 | 9 | 0.875 / 1 / 1 / 0.9167 |
| c | 11 | 19 | 0.157 | 1/19 | 11.79 | 3.96 | 4/19 | 6.08 | 88 | 59 | 17 | 8.5 | 0.8636 / 1 / 1 / 0.9091 |
| ac | 11 | 19 | 0.157 | 1/19 | 11.58 | 3.95 | 4/19 | 6.07 | 88 | 59 | 17 | 8.5 | 0.8636 / 1 / 1 / 0.9091 |
| ab-ignored | 12 | 21 | 0.15 | 1/21 | 11.16 | 4.23 | 4/21 | 3.64 | 88 | 59 | 17 | 9 | 0.875 / 1 / 1 / 0.9167 |
| bc-ignored | 12 | 21 | 0.15 | 1/21 | 11.36 | 4.24 | 4/21 | 3.64 | 88 | 59 | 17 | 9 | 0.875 / 1 / 1 / 0.9167 |
| abc-weak | 12 | 21 | 0.15 | 1/21 | 11.15 | 4.21 | 4/21 | 3.64 | 88 | 59 | 17 | 9 | 0.875 / 1 / 1 / 0.9167 |
| abc-ignored | 12 | 21 | 0.15 | 1/21 | 11.16 | 4.23 | 4/21 | 3.64 | 88 | 59 | 17 | 9 | 0.875 / 1 / 1 / 0.9167 |

**德累斯顿圣母教堂**（`frauenkirche-dresden`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 20 | 26 | 0.502 | 9/26 | 11.13 | 7.17 | 8/26 | 5.12 | 64.5 | 14.5 | 46 | 15.5 | 0.9722 / 0.8333 / 0.8333 / 1 |
| a | 20 | 26 | 0.503 | 9/26 | 10.71 | 7.2 | 8/26 | 5.12 | 64.5 | 14.5 | 46 | 15.5 | 0.9722 / 0.8333 / 0.8333 / 1 |
| b-weak | 22 | 27 | 0.497 | 9/27 | 14.77 | 8.11 | 8/27 | 5.02 | 64.5 | 16 | 44.5 | 19 | 0.7632 / 0.7895 / 0.7895 / 1 |
| b-ignored | 19 | 21 | 0.56 | 9/21 | 13.41 | 6.01 | 7/21 | 4.65 | 64.5 | 17.5 | 43 | 19 | 1 / 0.8 / 0.8 / 1 |
| c | 20 | 26 | 0.502 | 9/26 | 11.13 | 7.17 | 8/26 | 5.12 | 64.5 | 14.5 | 46 | 15.5 | 0.9722 / 0.8333 / 0.8333 / 1 |
| ac | 20 | 26 | 0.503 | 9/26 | 10.71 | 7.2 | 8/26 | 5.12 | 64.5 | 14.5 | 46 | 15.5 | 0.9722 / 0.8333 / 0.8333 / 1 |
| ab-ignored | 19 | 21 | 0.561 | 9/21 | 12.99 | 6.1 | 7/21 | 4.64 | 64.5 | 18 | 42.5 | 17 | 1 / 0.8 / 0.8 / 1 |
| bc-ignored | 19 | 23 | 0.541 | 10/23 | 12.89 | 5.7 | 7/23 | 4.65 | 64.5 | 17.5 | 43 | 18 | 1 / 0.8235 / 0.7647 / 1 |
| abc-weak | 22 | 27 | 0.498 | 9/27 | 14.58 | 8.18 | 8/27 | 5.02 | 64.5 | 16 | 44.5 | 18.5 | 0.7632 / 0.8421 / 0.7895 / 1 |
| abc-ignored | 19 | 23 | 0.541 | 10/23 | 12.48 | 5.7 | 7/23 | 4.64 | 64.5 | 18 | 42.5 | 16 | 1 / 0.8235 / 0.7647 / 1 |

每一组做了什么：

- **(a) 单开**：
  - 门本身只拦下 1 个模型（两次重复里是同一个：勃兰登堡门东南面，9 张里注册了 5 张，本来就没发布），从 `full` 降到 `prior_only`。评审担心的"同一个点拍的照片被当成已配准"，在这五栋的记录里就这一例。
  - 尺度误差没有变（|ln 尺度| 均方根 0.353 → 0.354，偏差超过 35% 的 16 个没少）。真实数据里尺度差得多的模型不是这个机制造成的。对存下的模型重算：尺度偏差超过 35% 的共 29 行（两次重复都算），GPS 内点中位 8 个，GPS 坐标的跨度中位 19.9 m，改在拟合出的布局上量是 17.8 m，都在 GPS σ 的三倍上下，哪一种量法都过门。偏差最大的是勃兰登堡门东北面一个 12 张的模型：10 个内点，跨度 24.7 m，尺度差 3.8 倍。29 行里 21 行是模型偏小。
  - 变的是 189 个 `full` 模型的偏航：GPS 那一边的 σ 变大了，融合时罗盘的份量跟着变大。118 行标定表里，偏航误差变小的 63 行，变大的 27 行，不变的 28 行；均方根 9.59° → 9.29°（−3.1%）。对的位姿多 1 个，错的少 1 个，超出自己 2σ 的 31 → 29 个。
  - 没有一条够得上"有用"的线（5%、2 个），也没有任何坏处。
- **(b) `weak`**：没标真北的读数按 45° 算。偏航变差：均方根 9.59° → 11.09°（+15.6%），偏航的 2σ 覆盖率 0.966 → 0.885，超出 2σ 的位姿 31 → 35 个。圣母教堂北面和南面各有一个本来就发布着的小模型（6 张、9 张，尺度本来就差 3 倍），偏航误差从 18° 和 22° 变成 33° 和 31°，过了"朝错面"的线（30°，`wrong_face_models_published` 0 → 2）。多发布了 3 个模型（比萨西南面 1 个，圣母教堂西面 2 个）。有坏处。
- **(b) `ignored`**：没标真北的读数不算。
  - 对的位姿多 3 个、错的少 3 个；比萨西南面一个 19–20 张的模型（原来被质量门的罗盘检查拦下，按参考偏航只差 5°）发布了。
  - 但圣母教堂西南面三个发布着的模型（24、34、39 张，按参考偏航差 7–9°、尺度差不到 12%）被质量门否掉，理由是"朝向和罗盘差 36–38°（中位数）"：这个扇区带罗盘的照片大多是磁北的，去掉以后中位数只剩一两个真北读数，而这一两个读数恰恰是错的。同一栋楼西面两个模型（7 张、10 张）反而发布了：7 张的偏航差 28°；10 张的只剩一个真北读数，它和 GPS 冲突，模型降成了 `prior_only`。
  - 超出 2σ 的位姿 31 → 33，看错面的相机多了。有坏处。
- **(c) 单开**：238 个模型一个都没变。记录里的偏航冲突（罗盘赢）共 12 个模型（两次重复都算），GPS 内点都不超过 5 个，裁决要求至少 8 个；其中能和参考比的 9 个，罗盘都是对的（偏航误差 < 15°），GPS 是错的。评审设想的"强 GPS 被一个坏罗盘否掉"在这五栋的记录里没有出现过；老规则在能核对的每一例里都判对了。
- **三个一起开，(b) 取 `weak`**：和 (b) `weak` 单开一样的坏处（偏航均方根 +13.8%，同样那两个模型过了朝错面的线）。
- **三个一起开，(b) 取 `ignored`**：
  - 对的位姿 142.5 → 149，错的 78.5 → 72；偏航差超过 15° 的模型 13 → 11 个；偏航中位数 4.91° → 4.22°；超出 2σ 的位姿 31 → 30。好处主要在勃兰登堡门（偏航均方根 6.91° → 5.88°，对的位姿多 2 个）和圣母教堂（对的多 3.5 个）。
  - (c) 在这里起了作用：(b) 去掉磁北读数以后，圣母教堂每次重复有 3 个模型（西南面 31 张的、西面 10 张的、42 张的全景模型）的罗盘先验只剩 1 个真北读数，和 GPS 差 40–49°，而 GPS 有 17、8、22 个内点。裁决把它们留给 GPS，回到 `full`；按参考，这三个模型跟 GPS 走以后偏航只差 3–6°，也就是说那个真北读数是错的。这正是评审说的"一个坏罗盘否掉强 GPS"，只是要先去掉磁北读数它才出现。西南面 31 张的和西面 10 张的那两个模型，有没有 (c) 都发布着；(c) 改的是它们的等级和朝向（见下面 `ab-ignored` 那一条）。
  - 代价也在圣母教堂，和 (b) `ignored` 单开时一样：西南面另外三个好模型被质量门否掉，西面两个差的发布了（7 张的偏航差 30°，离"朝错面"的线只差 0.3°；10 张的尺度差 45%）；这栋楼能和参考比的模型 26 → 23 个，|ln 尺度| 均方根 0.502 → 0.541，水平位置的覆盖率 0.833 → 0.765。总体上发布的模型数没变（68.5），尺度偏差超过 35% 的模型多 1 个（16 → 17），水平位置的覆盖率 0.932 → 0.915，都在事先写下的容许范围里。
- **(a) + (b) `ignored`，不开 (c)**（`ab-ignored`，补跑）：
  - 规则看的数和三个一起开几乎一样：对的位姿 149、错的 72、偏航差超过 15° 的模型 11 个、发布的模型 68.5，都相同；超出 2σ 的位姿 31（三个一起开是 30），偏航均方根 9.69°（9.59°），尺度偏差超过 35% 的模型 16/79（17/81），水平位置的覆盖率 0.930（0.915）。
  - 和三个一起开相比，变的只有圣母教堂，就是 (c) 裁决过的那 3 个模型（每次重复各 3 个，逐个模型数过：dev 两次重复共 238 个模型，裁决被问到 18 次，判给 GPS 6 次）。不开 (c)，它们的偏航归那一个真北读数，等级是 `prior_only`：
    - 西南面 31 张的（发布着）：模型和轮廓的夹角 18.75°；开 (c) 是 5.26°，老规则（`off`）是 6.27°（第一次重复的记录，下同）。
    - 西面 10 张的（发布着）：`prior_only`，没有可比的轮廓夹角；开 (c) 是 2.6°。
    - 42 张的全景模型（覆盖不够，没发布）：17.57°；开 (c) 是 5.39°，`off` 是 5.93°。
  - `prior_only` 的模型不算"能和参考比的模型"，所以这三个模型的朝向变差，规则里的误差数看不到；发布状态、对和错的位姿数没有变。
- **(b) `ignored` + (c)，不开 (a)**（`bc-ignored`，补跑）：对的位姿 +3、错的 −3，和 (b) `ignored` 单开一样；超出 2σ 的位姿 31 → 32，在容许范围里，但看错面的相机 3 → 3.5，有坏处。也就是说 (a) 是要的：加上它（就是 `abc-ignored`），洗礼堂多 1 个、勃兰登堡门多 2 个、圣母教堂多 0.5 个对的位姿，比萨第一次重复里看错面的相机从 7 台回到 6 台（和 `off` 一样）。
- **(a) + (c)，不开 (b)**（`ac`，补跑）：和 (a) 单开逐个数相同。没有 (b) 时，(c) 一个冲突也不判给 GPS（记录里冲突的模型，GPS 内点都不超过 5 个）。

按规则判：

| 组 | 有用 | 没坏处 | 结论 |
|---|---|---|---|
| a | 否（偏航均方根 −3.1%，对 +1、错 −1） | 是 | 不合格 |
| b-weak | 否 | 否（超出 2σ +4；朝错面的模型 +2、相机 +6.5；偏航覆盖率 −0.081） | 不合格 |
| b-ignored | 是（对 +3、错 −3） | 否（超出 2σ +2；看错面的相机 +0.5） | 不合格 |
| c | 否（什么都没变） | 是 | 不合格 |
| ac（补跑） | 否（和 a 相同） | 是 | 不合格 |
| ab-ignored（补跑） | 是（偏航差 > 15° 的模型 −2；对 +6.5、错 −6.5） | 是 | **合格** |
| bc-ignored（补跑） | 是（对 +3、错 −3） | 否（看错面的相机 +0.5） | 不合格 |
| abc-weak | 是（对 +2.5、错 −2.5） | 否（超出 2σ +1.5；朝错面的模型 +2、相机 +4；偏航覆盖率 −0.065） | 不合格 |
| abc-ignored | 是（偏航差 > 15° 的模型 −2；对 +6.5、错 −6.5） | 是 | **合格** |

合格的有两组：`ab-ignored` 和 `abc-ignored`。规则的最后一条说"几组都合格时选开关最少的那组，除非开关多的一组在'有用'的某一条上又好出同样的幅度"。把 `abc-ignored` 拿来和 `ab-ignored` 比（同样的算法，`ab-ignored` 放在对照的位置）：偏航均方根 −1.0%，|ln 尺度| 均方根 −0.6%，对和错的位姿数相同，偏航差超过 15° 的模型都是 11 个，尺度偏差超过 35% 的 16 → 17 个。"有用"的四条一条也不满足。所以按规则选的是 `ab-ignored`：(a) 和 (b) `ignored`；(c) 选不到。

当时的经过要照实记下：留出集的标准冻结时（501023e），只跑过前面七组，其中合格的只有 `abc-ignored`，所以提名的、在留出集上跑的是它。只开两个开关的组合，探索阶段在没提交的代码上跑过 `ac` 和 `bc-ignored`，`ab-ignored` 没有跑过，三组都没有从干净的提交跑、也没有提交；原来这里写的"只有三个一起开的这一组合格"只在那七组里成立。补跑是评审指出以后做的，只在 dev 上。

**默认值在这时候还没有动**（提交 501023e）。提名的这一组（`abc-ignored`）先在留出集上按下面的标准看一次，通过了才改默认值；在那之前，开关只通过重放的配置生效，产品的输出不变。下面的标准和结果是当时的原文。

**留出集的标准**（`eval/criteria/d34-v1.json`，和 dev 的结果同一个提交，写在看留出集任何一个打开开关的数之前）。从加进这个文件的那个提交跑三条命令，只跑一次：`d34-plain-test`（不重放的补算）、`d34-off-test`（对照）、`d34-abc-ignored-test`（提名的这一组；配置文件的 sha256 记在标准里，有测试守着）。别的组不在留出集上跑。

| 规则或核对 | 要求 |
|---|---|
| `control_reproduces_plain`（前提） | 留出集上 `off` 和 `plain` 的每个指标（风格化的用时和内存除外）、标定表的每一行都相同；对照组没有改任何模型 |
| `helps` | 提名组对 `off`，满足 dev 那条规则里"有用"的至少一条，同样的算法、同样的线 |
| `no_harm` | 提名组对 `off`，满足"没坏处"的每一条；两边都没有可比模型的覆盖率不判，四个都判不了算数据不足 |
| `stylization_not_worse` | 提名组的 `style_registration_wrong`、`style_qa_failed`、`struct_incorrect`、`struct_qa_failed`，每栋、每次重复都不高于 `off` |
| `wrong_face_models_published`（`compare` 判） | 提名组每栋、每次重复都是 0 |
| `photo_pose_wrong` 占 `photo_pose_gt_determined` 的比率（`compare` 判） | 合并和每栋都不比 `off` 差到超出抽样噪声（差值的 95% Newcombe 区间） |

两条 `compare` 规则用 `compare --criteria d34-v1 --noise-factor 0`（A 是 `d34-off-test`，B 是 `d34-abc-ignored-test`：两边打分的是同一批 SfM 产物）。全部通过：三个开关成为默认值，`RECON_OUTPUT_VERSION` 升到 3。有一条不过，或者数据不足：默认值都保持关，代码和开关留着。不管结果怎样，这次留出集的运行都保留、都报告，之后不再调参数。

**留出集的结果：通过**。三条命令从 501023e 的干净工作树跑（2026-11-12，只跑了这一次），代码和 dev 那八条用的逐字节相同（结果里的 `code_sha256` 一样）。结果文件在 `eval/results/commons-ten-v1/d34-regeo/`（`d34-plain-test.json`、`d34-off-test.json`、`d34-abc-ignored-test.json`）。

| 规则或核对 | 结果 | 数 |
|---|---|---|
| `control_reproduces_plain` | PASS | 留出集两次重复共 144 个模型，都重放了，最大相机差 0.0000 m；对照组没有改任何模型；`off` 对 `plain` 的 1590 个值都相同（另有 22 个风格化用时和内存的值不同），标定表 44 行都相同 |
| `helps` | PASS | 三条都满足：偏航均方根 7.26° → 6.35°（−12.5%）；错的位姿 34.5 → 31.5（少 3 个），对的 62 → 65（多 3 个） |
| `no_harm` | PASS | 发布的模型 25.5 → 26.5；显示的位姿 179.5 → 179.5；超出自己 2σ 的 12 → 11.5；看错面的模型和相机没有增加；覆盖率 偏航 1 → 1、尺度 1 → 1、水平 0.909 → 0.913、倾斜 0.955 → 0.957；尺度偏差超过 35% 的模型 5 → 5，偏航差超过 15° 的 2 → 1 |
| `stylization_not_worse` | PASS | 五栋、两次重复的四个指标（40 个值）都不高于对照 |
| `wrong_face_models_published` 为 0 | PASS | 五栋都是 0 |
| 错位姿的比率不比对照差 | PASS | 合并 34.5/96.5 = 0.358 → 31.5/96.5 = 0.326；每栋都不差（悉尼歌剧院没有参考位姿，没有比率） |
| `no_other_arm_on_held_out` | 如实 | 留出集上只有 `plain`、`off`、`abc-ignored` 三个结果 |

**总体**

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 25.5 | 28 | 0.248 | 5/28 | 7.26 | 3.37 | 2/28 | 3.82 | 179.5 | 62 | 34.5 | 12 | 1 / 1 / 0.9091 / 0.9545 |
| abc-ignored | 26.5 | 29 | 0.246 | 5/29 | 6.35 | 3.42 | 1/29 | 3.82 | 179.5 | 65 | 31.5 | 11.5 | 1 / 1 / 0.913 / 0.9565 |

**配准等级，和重放改了什么**（两次重复都算）

| 组 | 模型 | 重放了 | 原样留下 | 变换变了 | 等级变了 | 发布状态变了 | full / prior_only / none | 已发布 |
|---|---|---|---|---|---|---|---|---|
| off | 144 | 144 | 0 | 0 | 0 | 0 | 90 / 54 / 0 | 51 |
| abc-ignored | 144 | 144 | 0 | 92 | 2 | 2 | 92 / 52 / 0 | 53 |

**按模型注册的照片数分档**（每格：模型数，|ln 尺度| 均方根，尺度偏差 > 35% 的个数，偏航均方根 °）

| 组 | 3-6 张 | 7-10 张 | 11-20 张 | 21+ 张 |
|---|---|---|---|---|
| off | 0，—，0/0，— | 8，0.195，2/8，7.56 | 19，0.272，3/19，7.3 | 1，0.103，0/1，2.5 |
| abc-ignored | 0，—，0/0，— | 9，0.185，2/9，7 | 19，0.275，3/19，6.16 | 1，0.103，0/1，2.5 |

**圣家堂**（`sagrada-familia`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 15 | 16 | 0.3 | 5/16 | 9.29 | 7.32 | 2/16 | 3.7 | 52 | 38 | 8 | 2 | 1 / 1 / 1 / 0.9333 |
| abc-ignored | 16 | 17 | 0.294 | 5/17 | 7.98 | 5.12 | 1/17 | 3.82 | 52 | 41 | 5 | 2 | 1 / 1 / 1 / 0.9375 |

**匈牙利国会大厦**（`hungarian-parliament`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 4.5 | 4 | 0.245 | 0/4 | 2.97 | 1.72 | 0/4 | 16.49 | 38.5 | 3.5 | 10 | 5 | 1 / 1 / 0 / 1 |
| abc-ignored | 4.5 | 4 | 0.245 | 0/4 | 2.97 | 1.72 | 0/4 | 16.49 | 38.5 | 3.5 | 10 | 5 | 1 / 1 / 0 / 1 |

**林肯纪念堂**（`lincoln-memorial`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 2 | 3 | 0.104 | 0/3 | 3.62 | 3.8 | 0/3 | 8.82 | 31 | 4.5 | 15.5 | 4 | 1 / 1 / 1 / 1 |
| abc-ignored | 2 | 3 | 0.104 | 0/3 | 3.3 | 3.42 | 0/3 | 8.82 | 31 | 4.5 | 15.5 | 3.5 | 1 / 1 / 1 / 1 |

**印度门**（`india-gate`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 4 | 5 | 0.052 | 0/5 | 1.99 | 2.22 | 0/5 | 2.75 | 48 | 16 | 1 | 1 | 1 / 1 / 1 / 1 |
| abc-ignored | 4 | 5 | 0.052 | 0/5 | 1.98 | 2.22 | 0/5 | 2.75 | 48 | 16 | 1 | 1 | 1 / 1 / 1 / 1 |

**悉尼歌剧院**（`sydney-opera-house`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 0 | 0 | — | 0/0 | — | — | 0/0 | — | 10 | — | — | 0 | — / — / — / — |
| abc-ignored | 0 | 0 | — | 0/0 | — | — | 0/0 | — | 10 | — | — | 0 | — / — / — / — |

留出集上要照实说的几点：
- 好处几乎都在圣家堂一栋：偏航均方根 9.29° → 7.98°、中位数 7.32° → 5.12°，对的位姿多 3 个、错的少 3 个；南面一个 20 张里注册了 10 张的模型从 `prior_only` 变成 `full` 并且发布了（两次重复都是这一个）。另外四栋基本没变：匈牙利国会大厦的数一个都没动，印度门只有偏航均方根从 1.99° 变成 1.98°，林肯纪念堂偏航均方根 3.62° → 3.30°，悉尼歌剧院在记录里一个模型都没有。
- 尺度误差没有变（|ln 尺度| 均方根 0.248 → 0.246，偏差超过 35% 的还是 5 个）。评审第一条发现指望 (a) 改善尺度，dev 和留出集都没有看到。
- 能和参考比的模型在留出集上只有 28 个（dev 是 83 个），"偏航差超过 15° 的模型少 2 个"这一条在留出集上没有满足（2 → 1），通过靠的是另外三条。
- 偏航裁决 (c) 在留出集上一次也没有裁决过。评审之后逐个模型数了一遍（从 501023e 的干净工作树，用提名组的配置把留出集两次重复的 144 个模型重放一遍，只数裁决，不和参考位姿比，不出任何指标）：144 个都重放了；"GPS 拟合可用、又和偏航先验冲突"共 4 次（2 次是罗盘先验，2 次是伪罗盘），判给 GPS 的 0 次，模型 JSON 里 `anchor.raw.yaw_prior_overruled` 为真的 0 个。圣家堂南面那个模型变成 `full`，是因为去掉磁北读数以后先验换成了伪罗盘，和 GPS 不再冲突。
- 所以这次留出集的重放，同时就是 `ab-ignored` 的重放：一个先验没被否掉的模型，(c) 开着关着出来的东西逐位相同（`tests/test_georegister.py` 有测试守着）。**留出集的通过是 (a) + (b) `ignored` 的证据；对 (c)，它既不是支持也不是反对。** 留出集上没有为此再跑任何一组。

**决定**（评审之后更正的）：默认值是 `ab-ignored` 这一组：`gps_span_model_side` 打开，`compass_other` = `ignored`，`yaw_arbitration` **关**。

- 按 dev 的规则，合格的组里开关最少的是它；(c) 加上去够不上"有用"的任何一条；留出集的通过也只说明它。
- 6f9e8cf 曾经把三个开关都设成默认值（`RECON_OUTPUT_VERSION` 3）。评审之后 (c) 改回关，版本升到 4；金样只有版本号那一行变了（金样场景里没有偏航冲突）。(c) 的代码和开关留着，配置 `eval/configs/d34-regeo/abc-ignored.json` 就是把它打开。
- 这不是看了留出集的数以后调的：(c) 在留出集上没有裁决过，开着关着留出集的每个数都一样；改它依据的是 dev 上补跑的三组和事先写下的规则。标准文件 `eval/criteria/d34-v1.json` 冻结时的字段一个没动，后面加了一段 `addendum` 记这件事。
- 关着 (c) 的代价，在 dev 上看得见，规则的数看不见：圣母教堂西南面 31 张的那个发布着的模型，偏航归一个错的真北读数，和轮廓的夹角 18.75°（老规则 6.27°，开 (c) 5.26°）；西面 10 张的那个发布着的模型是 `prior_only`。这是 (b) 去掉磁北读数以后才出现的"一个读数否掉 17 个 GPS 内点"。
- 不开 (c) 的理由除了规则，还有两条：它的四个阈值（8 个内点、5 倍 σ、3 个读数、2 倍散布）没有在真实数据上调过，dev 上那三个模型的冲突是 40.3–40.4°、40.8° 和 49.3°，前两个刚过 40° 的线；合成场景里，一两个**对的**真北读数碰上一个转了 90° 的 GPS 布局（20 个定位，比如地理标签是沿着另一条街手工标的）时，裁决 40 个种子里 40 个都跟 GPS 走，模型转错 80° 以上（中位 91°），老规则都是对的（评审的模拟，这里重做了一遍，写成了测试）。要打开它，需要它自己的证据。

之后没有再调任何参数。这样定下来的默认值还缺一样证据：重放看不到选候选和重试（6.4 节），现在这两处也只看真北读数（有单元测试守着这两处读的是什么，`tests/test_georegister.py`，但它在真实照片上的效果没有量过）；下一次全新运行会把这一部分一起量进去，在那之前不要拿新默认值的结果和 D31 的全新运行直接比。（这次全新运行做了：6.14 节。）旧的行为（三个开关都关）仍然可以用配置 `eval/configs/d34-regeo/off.json` 重放出来。

**评审之后的更正**（2026-11-12，这一节改了哪些话）：

| 原来写的 | 哪里不对 | 现在 |
|---|---|---|
| "只有三个一起开的这一组合格"，"三个开关成为默认值" | 只在当时跑过的七组里成立。`ab-ignored` 没有跑过；补跑以后它自己合格，而且按规则该选它 | 判定表加了三行；默认值是 (a) + (b) `ignored`，(c) 关 |
| "提交进仓库的七组里没有只开两个开关的组合"（另两处文档写的是"没有跑"） | 探索阶段在没提交的代码上跑过 `ac` 和 `bc-ignored`（笔记里有），没有从干净的提交跑，也没有提交；`ab-ignored` 确实没跑过 | 三组都从 65781fd 的干净工作树跑了，结果在 `d34-regeo/` 里 |
| "留出集确认的是三个开关合在一起" | (c) 在留出集上一次没有裁决过，三个一起开和只开 (a)、(b) 在留出集上是同一个结果 | 留出集的通过只是 (a) + (b) 的证据 |
| "西南面 31 张的那个模型因此仍然发布着" | 有没有 (c) 它都发布着 | (c) 改的是它的等级和朝向 |

补跑和核对用的三个小脚本（按规则判定、数裁决、生成 `addendum`）在开发机的草稿目录里，没有进仓库；规则的算法写在标准文件的 `dev_rule` 里，判定的结果和用到的每个数在 `addendum` 里。

**评审第二轮之后补记**（2026-11-12；这一节的数没有变，代码和结果文件也没有动）：

- 重放怎么读 D34 之前记录的运行（摘要里没有三个开关、照片表没有 `heading_ref` 列），原来没有测试守着，现在有了（`tests/test_eval_real.py`，[D34 的记录](decisions/D34-review-fixes.md)第 6 步"评审第二轮之后"）。这件事要紧，是因为它错了不出声：把"从摘要取标记"这一处改坏，在 dev 的记录上按 `ab-ignored` 重放，报告照旧是"238 个模型都重放了，最大相机差 0.0000 m"，其实 182 个模型和正确的重放不一样（偏航先验是罗盘的模型全部换成了伪罗盘），圣母教堂两次重复多发布 6 个模型。
- 用现在的代码按 `ab-ignored` 和按运行自己的参数把 dev 的 10 个建筑运行重放了一遍（直接调 `replay_run`，不和参考位姿比）：报告（模型数、重放了几个、变换变了几个、各等级和发布的模型数）都和本节用的 `d34-ab-ignored-dev.json`、`d34-off-dev.json` 里记的相同。
- 本节带 `ignored` 或 `weak` 的各组，重放里的质量门比全新的运行多查了标成真北的照片（6.4 节）。在 dev 上量过 `ab-ignored` 这一组（别的组没有量）：238 个模型里质量门的结论、理由和发布状态 0 个不同，只有比萨第二次重复一个没发布的模型里，一台本来就带着别的标记的相机多了一个 `facing_away`；那次重复两遍重放出的指标、显示的位姿和地图都相同，本节的数不受影响。留出集这次没有跑，评审数的是 144 个模型里 0 处不同。重放的代码没有为此改动。

### 6.14 D34 的全新运行（第四个开关，和地理配准默认值的全新确认）

6.13 节的默认值是离线重放定的。重放有两样东西量不了，这一节用全新的 SfM 来量：

- 第四个开关 (d) `Settings.facing_away_warns_other_refs`：罗盘朝向背对建筑、又没有标成真北的照片，照收并带警告 `heading_away_unverified`，不再按 `facing_away` 拒绝。它改变收下哪些照片，重放不出来。
- (a) + (b) 这两个默认值在全新运行里的效果：建图时选哪个候选、要不要重试，现在只看真北读数，重放看不到这两处。(c) 不在这次要判的范围里，保持关。

标准文件是 `eval/criteria/d34b-v1.json`。**下面"事先写好的"这一部分和标准文件、配置文件同一个提交，写在任何一次 D34 全新运行之前**（2026-11-12，从 5fa4619 出发）；dev 的结果和留出集用哪个配置，在留出集运行之前补进标准文件的 `dev_outcome`（提交 6213160）；留出集的结果在它之后补进 `held_out_outcome`（提交 776b0e7）；评审之后在 `held_out_outcome` 里加了一段 `addendum`，别的字段没有动；看到数以后不改规则。结果在本节后面。

#### 事先写好的：比什么

- **B（全新运行）**：`eval/configs/d34-heading-warn.json`，就是产品默认值加上 (d) 打开，别的一样。和 D31 的全新运行同样的做法：每栋跑两次，`--time-cap 3600`，风格化开，一次只跑一栋；dev 五栋带 `--max-requests brandenburg-gate=104 frauenkirche-dresden=31`。结果是 `eval/results/commons-ten-v1/d34-dev.json` 和 `d34-test.json`。参考缓存用 D31 全新运行和 6.13 节重放读的那一份的副本（印度门第 2 版）。
- **A（基线）**：`d30-a-dev.json`、`d30-a-test.json`。它们就是 D31 的全新运行（`d31-dev`、`d31-test`）在 716de16 上补算到指标第 8 版的结果，带着那两次运行自己的噪声块。`compare` 不肯直接比 `d31-<划分>.json`（指标第 7 版）。A 的地理配准是记录下来的老规则，`facing_away` 不分标记一律拒绝。
- **A′（只报告）**：`d34-regeo/d34-ab-ignored-dev.json`、`d34-regeo/d34-abc-ignored-test.json`，同一批 D31 的 SfM 产物按 (a) + (b) 重放的结果（留出集上 (c) 一次没裁决过，那一组等于 (a) + (b)）。B 和 A′ 的差别是 (d)、SfM 重跑一遍的差别、以及重放看不到的那部分。
- B − A 里混着 (a)、(b)、(d) 和 SfM 的运行间差别。用两样东西把它们分开：
  - **全新运行自己的配对重放**（核对 `fresh_pair`）：把 B 这次运行保留的稀疏模型按老的地理配准（`eval/configs/d34-regeo/off.json`）重放一遍，再按运行自己的参数重放一遍。两遍用的是同一批 SfM 产物、同一批收下的照片，差别只来自 (a) + (b)，没有运行间噪声。
  - **新收下的照片逐张的去向**（核对 `newly_accepted_photos`）：带 `heading_away_unverified` 警告的照片，注册进了哪个模型、被标记没有、显示的位姿对不对。

#### 事先写好的：(d) 会多收多少照片

照片校验不依赖 SfM，所以可以从 D31 运行记下的照片行直接算：只因为 `facing_away` 被拒、而且朝向没有标成真北的照片，(d) 打开以后会收下。两次重复的数相同。

| 建筑 | 划分 | 上传 | D31 收下 | D31 按 `facing_away` 拒的（其中标成真北的 / 还有别的拒绝原因的） | (d) 新收下 | 预计收下 |
|---|---|---|---|---|---|---|
| 洗礼堂 | dev | 56 | 51 | 5（2 / 0） | 3 | 54 |
| 亚琛 | dev | 91 | 75 | 16（16 / 0） | 0 | 75 |
| 勃兰登堡门 | dev | 149 | 136 | 8（4 / 1） | 3 | 139 |
| 比萨斜塔 | dev | 124 | 115 | 6（5 / 0） | 1 | 116 |
| 圣母教堂 | dev | 88 | 83 | 5（0 / 0） | 5 | 88 |
| **dev 合计** | | 508 | 460 | 40 | **12** | **472** |
| 圣家堂 | test | 150 | 128 | 20（5 / 0） | 15 | 143 |
| 匈牙利国会大厦 | test | 97 | 86 | 6（2 / 0） | 4 | 90 |
| 林肯纪念堂 | test | 77 | 58 | 15（12 / 3） | 2 | 60 |
| 印度门 | test | 99 | 65 | 11（1 / 4） | 6 | 71 |
| 悉尼歌剧院 | test | 90 | 62 | 14（10 / 1） | 3 | 65 |
| **留出集合计** | | 513 | 399 | 66 | **30** | **429** |

要先说清楚的一点：(d) 管得到的照片很少，dev 每次运行 12 张（2.6%），留出集 30 张（7.5%，一半在圣家堂）。按 `facing_away` 拒掉的 106 张里，64 张不归它管（57 张标的是真北，另有 7 张没标真北、但同时还有别的拒绝原因）。所以合计的指标多半看不出它的效果，要看那几十张照片各自的去向。

#### 事先写好的：规则（`compare --criteria d34b-v1`，A 对 B）

用 `compare` 文档里的噪声处理（6.2 节、第 8 节）：差值超过 max(该指标的最小变化量, 相对下限 × A, 1 × 记录的运行间极差) 才算显著；噪声来源是 A、B 自己和 `eval/results/commons-ten-v1/` 下同一指标版本的结果（今天是 `d30-a-dev`、`d30-a-test`；`d34-dev` 提交以后也算）。下面的门槛是今天算得出的，B 自己的运行间极差只会把门槛抬高，不会压低。

| 规则（都是守卫） | 要求 | dev：A，过不了的 B | 留出集：A，过不了的 B |
|---|---|---|---|
| `wrong_face_models_published` | 每栋、每次重复都是 0 | 0，> 0 | 0，> 0 |
| `style_registration_wrong`、`style_qa_failed`、`stale_overwrites` | 每栋、每次重复都是 0 | 0，> 0 | 0，> 0 |
| `struct_incorrect`、`struct_qa_failed` | 同上；没有风格化模型的建筑（A 里的悉尼歌剧院）不判，点名留出 | 0，> 0 | 0，> 0 |
| `models_published` | 划分的汇总不显著变少 | 68.5，< 67.5 | 25.5，< 24.5 |
| `facades_published` | 同上 | 20，< 20 | 9，< 9 |
| `photo_pose_determined`（有显示位姿的照片） | 同上 | 317，< 304 | 179.5，< 166.5 |
| `wrong_face_cameras_shown` | 汇总和每一栋都不显著变多 | 汇总 3，> 9；比萨 3，> 9；其余四栋 0，> 0 | 汇总 0，> 6；五栋都是 0，> 0 |
| `gt_gross_shown` | 同上 | 汇总 1.5，> 2.5；勃兰登堡门 1，> 1；比萨 0.5，> 1.5；其余 0，> 0 | 汇总 4.5，> 5.5；匈牙利国会大厦 4，> 4；林肯纪念堂 0.5，> 1.5；其余 0，> 0 |
| `photo_pose_wrong` 占 `photo_pose_gt_determined` 的比率 | 合并和每一栋都不比 A 差到超出抽样噪声（差值的 95% Newcombe 区间） | 78.5/221 = 0.355 | 34.5/96.5 = 0.358 |
| `geo_scale_error_median_pct` | 汇总不显著变大 | 11.81，> 14.90 | 12.19，> 15.28 |
| `geo_yaw_error_median_deg` | 同上 | 5.00，> 5.50 | 2.99，> 3.49 |

没有代价规则：同样的工作在这台机器上隔一天会多用约 1.5 倍 CPU（6.12 节），比多收 12–30 张照片的影响大得多；`cpu_time_s` 只报告。

`wrong_face_cameras_shown` 和 `gt_gross_shown` 逐栋也判（和 `d31-v1` 一样）：(d) 是往某几栋楼里加照片，合计的数会把一栋楼的问题盖住。两次重复估出来的噪声偏小（第 8 节），所以这两条在记录的极差是 0 的楼上很紧：多一台就不过。守卫不过就是不过，不事后放宽。

#### 事先写好的：核对（`compare` 之外判）

| 核对 | 角色 | 要求 |
|---|---|---|
| `runs_comparable` | 前提 | B 的每栋每次重复都跑完，没有被时间上限截断，上传请求数和 A 一样；`compare` 没有列出输入不同的建筑；B 是从干净的提交、用冻结的配置跑的。不满足的建筑点名，这一对算证据不足，不算不过 |
| `accepted_photos_rise` | 守卫 | 划分的 `photos_accepted` 变多（记录的噪声是 0，多一张也算）；每栋每次重复的漏斗等于上面的预计：收下的张数、带 `heading_away_unverified` 警告的张数、`facing_away` 的张数、别的拒绝原因的张数。对不上的逐张列出；除非合计还是变多、而且每处差别都是"和新收下的某张照片重复"，否则不过 |
| `shown_pose_coverage` | 守卫 | `photo_pose_uncertainty_coverage`（有参考位姿的显示位姿里，落在自己报的 2σ 范围内的比例，划分合并）不显著变低：`compare` 给这个划分的那一行不是"差值为负而且显著"。这个指标在 `compare` 里没有方向，规则判不了，所以放在核对里，门槛算法一样（今天：dev 0.860，低于 0.755 不过；留出集 0.876，低于 0.771 不过） |
| `georeg_errors` | 守卫 | 按 `georeg` 的算法（能和参考比的模型，两次重复里同一个只算一次）：|ln 尺度| 的均方根和偏航的均方根，B 不比 A 高出 max(A 的 5%, A 两次重复之间的差, B 两次重复之间的差)；尺度偏差超过 35% 的模型所占比例、偏航差超过 15° 的模型所占比例，不比 A 差到超出抽样噪声（`benchmark.rate_worse`）。A：dev 83 个模型，0.353、9.59°，16/83、13/83，两次重复之间差 0.000、0.33°；留出集 28 个模型，0.248、7.26°，5/28、2/28，两次重复之间差 0.007、0.08° |
| `newly_accepted_photos` | 守卫 | 带 `heading_away_unverified` 警告收下的照片（每次重复都算）：没有一张是"朝错面"（偏航差 ≥ 30°）的已发布模型里没被标记的相机；它们显示出来的 SfM 位姿里有参考位姿的不少于 10 个时，错的比例不比别的照片的 SfM 位姿差到超出抽样噪声；不到 10 个时这一半只报告、不判。逐栋列出：收下几张、注册进任何模型的、注册进已发布模型的、在里面被标记的（哪种标记）、按 SfM 显示的、只按位置显示的、对 / 错 / 没有参考 |
| `fresh_pair` | 守卫 | 把 `d34-v1` 照它自己的写法用在全新运行上。A = B 这次运行按 `d34-regeo/off.json` 重放，B = 同一次运行按自己记录的参数重放（`--replay default`）：`d34-v1` 的两条规则（`compare --criteria d34-v1 --noise-factor 0`）和"有用""没坏处""风格化不变差"三项核对都通过，算法和线同 6.13 节；`d34-v1` 的前提"对照组等于不重放的补算"在这里换成"按运行自己的参数重放，等于这次运行不重放的补算"（每个指标、标定表每一行都相同，没有模型变了变换、等级或发布状态）。三个结果放在 `eval/results/commons-ten-v1/d34-fresh/`（子目录：不进排行榜，不算噪声来源） |
| `side_by_side` | 只报告 | A、A′、B 的 `georeg` 表（总体和每栋）；A′ 对 B 的 `compare`；每栋收下的照片、发布的模型和立面、显示位姿的对错、墙离轮廓多远、哪些楼仍然一个模型都没有 |

#### 事先写好的：怎么定

**dev（可以调）**。A = `d30-a-dev`，B = `d34-dev`。

- 守卫规则和守卫核对在 dev 上没有一条不过，(d) 才被提名，留出集用 `d34-heading-warn` 跑。dev 上证据不足的规则或核对点名列出，不拦。
- 有守卫不过：(d) 不提名，留出集用默认配置（`default.json`，(d) 关）跑，只当作 (a) + (b) 的全新确认。有一个例外，只凭 dev 的数决定：可以再用默认配置跑一次全新的 dev，如果打开 (d) 的那次对它能通过同样的守卫（说明先前不过不是 (d) 造成的），(d) 仍然提名。
- dev 上对噪声多留一处余地：按张数、个数求和的指标上，一条"不显著变差"的守卫按 `compare` 的门槛不过，但变化没有超出 dev 各栋记录的极差之和（`compare` 给分划分汇总用的那个上界，`noise_bound`），不算 (d) 的问题，点名并把两个数都列出来。原因是 dev 汇总的极差可能是 0，而两栋楼各自在两次重复之间差 1（`d30-a-dev` 的 `facades_published`：亚琛和比萨一增一减）。留出集上没有这处余地。
- dev 上的 `fresh_pair` 和 A′ 只报告：(a)、(b) 本来就是在这五栋上选的。

**留出集（冻结，只跑一次）**。从带着 dev 结果和提名的那个提交跑：一次全新运行（留出集唯一的一次 COLMAP），然后三条不跑 COLMAP 的补算（不重放、按自己的参数重放、按老规则重放），再按两个标准文件机械地判。

- `verdict_d`（只在 (d) 被提名时判）：A = `d30-a-test`、B = `d34-test`，本文件的守卫规则和守卫核对都通过、前提成立，是 PASS；有一条不过是 FAIL；没有不过但有缺数据的是 INSUFFICIENT。
- `verdict_georegistration`：`d34-v1` 用在全新运行的配对重放上（`fresh_pair`），按 `d34-v1` 自己对结论的定义。
- 默认值：
  - `verdict_georegistration` 通过：`gps_span_model_side` 和 `compass_other` = `ignored` 保持默认。不过或者证据不足：两个都改回老的行为（它们在 dev 上是一起才合格的），`RECON_OUTPUT_VERSION` 加 1，(d) 不管自己的结论如何也保持关（它是在这两个开着的时候量的）。
  - `verdict_d` 和 `verdict_georegistration` 都通过：`facing_away_warns_other_refs` 改成默认打开，`RECON_OUTPUT_VERSION` 加 1。`verdict_d` 不过、证据不足，或者 (d) 没被提名：保持关。
  - `verdict_d` 不过而 `verdict_georegistration` 通过：把不过的守卫背后的模型和位姿逐个列出，写明里面有没有新收下的照片。这时 (a) + (b) 里重放看不到的那一部分（选候选、重试）和 (d) 分不开，照实写成"没有分开"。
  - `yaw_arbitration` 不在这次要判的范围里，保持关。
  - 留出集的每个结果都保留、都报告；之后不调任何参数。

#### dev 的结果：(d) 没有被提名

两次全新的 dev 运行都从 e3928f0 的干净工作树跑（2026-11-12），代码逐字节相同（`code_sha256` e4fc7a57…），重建输出版本 4，每栋两次重复，都跑完了，没有被时间上限截断：

- `d34-dev`：配置 `d34-heading-warn`（(d) 打开），按冻结的命令。结果 `eval/results/commons-ten-v1/d34-dev.json`。
- `d34-dev-doff`：配置 `default`（(d) 关），是事先写好的那条例外用的（见下面）。结果放在 `eval/results/commons-ten-v1/d34-fresh/d34-dev-doff.json`：冻结的标准只把 `d34-dev` 列为留出集那一对的噪声来源，没有列这一次，所以放在 `compare` 不读噪声的子目录里；它自己当 A 的那一对照样用它的噪声块。这次运行里洗礼堂的第一次重复删掉重跑过一遍（`--resume` 只重做这一次重复）：它和另一个只跑了约一分半钟的 COLMAP 作业重叠过，违反"一次只跑一个 COLMAP"。重跑前后 7 个作业注册的张数、发布与否、等级都相同，5 个作业的点数差 1–19 个；提交的结果是重跑以后的。

**(d) 多收的照片和预计的一样**：每次运行 12 张，460 → 472（洗礼堂 +3、勃兰登堡门 +3、比萨 +1、圣母教堂 +5、亚琛 0），两次重复相同。

**更正一处事先写错的预计**（记在标准文件 `dev_outcome.erratum`，冻结的字段没动）。上面"(d) 会多收多少照片"那张表里收下的张数是对的，但标准文件的 `predicted_acceptance` 另有两个数算错了：一张照片既背对建筑、又有别的拒绝原因、朝向又没标真北时，(d) 打开以后它照样被拒，但拒绝原因里不再有 `facing_away`，改带警告 `heading_away_unverified`。所以打开 (d) 以后，按 `facing_away` 拒的只剩标成真北的那些，带这个警告的是所有没标真北的背对照片（不只是新收下的）。dev 上只有勃兰登堡门一张（同时是 `not_subject`）：`facing_away` 实际 4 张（原来写的 5），带警告的 4 张（原来写的 3）。留出集上受影响的是林肯纪念堂（12、3；原来写 13、2）、印度门（1、10；原来写 5、6）、悉尼歌剧院（10、4；原来写 11、3）。更正后的数是从 D31 运行的照片行重新算的，十栋都对得上，有测试守着；它只改预计的张数，不放宽任何一条守卫。核对 `accepted_photos_rise` 按更正后的数判是通过；按冻结时的数判是不过（就差勃兰登堡门这一张），两个结论都记在标准文件里。

**冻结的那一对（A = `d30-a-dev`，B = `d34-dev`）**：

| 守卫规则 | A | B | `compare` 的结果 | 说明 |
|---|---|---|---|---|
| `wrong_face_models_published` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `style_registration_wrong` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `style_qa_failed` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `struct_incorrect` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `struct_qa_failed` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `stale_overwrites` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `models_published` | 68.5 | 66 | 不过 | significantly worse: aggregate；汇总变化 -2.5，`compare` 的门槛 1，各栋记录的极差之和 3；按 dev 的噪声余地不算在 (d) 头上 |
| `facades_published` | 20 | 18 | 不过 | significantly worse: aggregate；汇总变化 -2，`compare` 的门槛 0，各栋记录的极差之和 2；按 dev 的噪声余地不算在 (d) 头上 |
| `wrong_face_cameras_shown` | 3 | 1 | 通过 | better, delta -2 (threshold 6) |
| `gt_gross_shown` | 1.5 | 3.5 | 不过 | significantly worse: aggregate, brandenburg-gate, frauenkirche-dresden；汇总变化 +2，`compare` 的门槛 1，各栋记录的极差之和 1；**算在 (d) 头上** |
| `photo_pose_determined` | 317 | 308.5 | 通过 | worse, delta -8.5 (threshold 13) |
| `photo_pose_wrong` | 0.3552 | 0.2929 | 通过 | pooled 78.5/221 = 0.355 -> 61.5/210 = 0.293 (95% end -0.149); nearest building: brandenburg-gate 9.5/66 = 0.144 -> 8/68 = 0.118 (95% end -0.145) |
| `geo_scale_error_median_pct` | 11.8137 | 12.5688 | 通过 | worse, delta +0.7551 (threshold 3.0875) |
| `geo_yaw_error_median_deg` | 5.0038 | 3.9413 | 通过 | better, delta -1.0625 (threshold 0.5) |

| 核对 | 结果 | 数 |
|---|---|---|
| `runs_comparable` | 通过 | 5 buildings x 2 repeats ran to the end with A's request counts, inputs identical, commit e3928f0 clean, config d34-heading-warn (d_on), code_sha256 e4fc7a57bf40 |
| `accepted_photos_rise` | 通过 | split 460 -> 472 (delta +12, threshold 0); every funnel equals the corrected prediction: True; florence-baptistery 51 -> 54, aachen-cathedral 75 -> 75, brandenburg-gate 136 -> 139, leaning-tower-pisa 115 -> 116, frauenkirche-dresden 83 -> 88 |
| `shown_pose_coverage` | 通过 | 0.8597 -> 0.8857 (delta +0.026, threshold 0.1049, noise 0.1049) |
| `georeg_errors` | 通过 | checked models 83 -> 94; scale_rms_ln 0.353 -> 0.375 (margin 0.047: 5 % of A 0.01765, A r1/r2 0, B r1/r2 0.047) ok; yaw_rms_deg 9.59 -> 8.93 deg (margin 0.4795: 5 % of A 0.4795, A r1/r2 0.33, B r1/r2 0.16) ok; scale_off 16/83 = 0.193 -> 20/94 = 0.213 (95% end -0.101) ok; yaw_off 13/83 = 0.157 -> 14/94 = 0.149 (95% end -0.117) ok |
| `newly_accepted_photos` | 通过 | 朝错面那一半 PASS；比例那一半 not judged（它们按 SfM 显示、有参考的 6 个张次，错 0 个；别的照片 388 个里错 111 个） |

算在 (d) 头上的守卫：`gt_gross_shown`；按噪声余地放过的：`models_published`、`facades_published`；这一对不通过。

三条守卫按 `compare` 的门槛不过。`models_published` 和 `facades_published` 的变化没有超出 dev 各栋记录的极差之和（事先写好的那处噪声余地），不算在 (d) 头上。`gt_gross_shown`（正在显示的已发布模型里，没被标记、离参考位姿超过 20 m 或 45° 的相机）不在余地之内：汇总 1.5 → 3.5，各栋极差之和只有 1；单看勃兰登堡门（1 → 2.5，门槛 1）和圣母教堂（0 → 1，门槛 0）也都显著变多。按事先写好的规则，这一条算在 (d) 头上，(d) 不提名，除非例外替它澄清。

噪声余地那句话按字面有两种读法：只管要求写的是 `not_worse` 的规则；或者 `not_worse_per_building` 的规则里汇总的那一半也管。单栋自己显著变差的，两种读法都不放过（单栋是和它自己记录的极差比的，没有什么可加总）。判定用严的那一种；两对上两种读法的结论都相同，标准文件里两个都记了。

**例外：再跑一次不开 (d) 的全新 dev，拿它当 A**（事先写好的那一条；`d34-dev-doff`，2026-11-12 08:00–10:16，五栋都跑完，8,159 秒）。一点说明：这次运行的洗礼堂第一次重复，开头有 85 秒和一个单作业的 COLMAP 诊断同时在跑，所以运行结束以后把这一次重复删掉、用同一条命令加 `--resume` 单独重跑了（别的重复原样复用），结果文件里的 `provenance.started` 和 `wall_s` 因此是重跑那一次调用的。重跑前后这次重复的 7 个作业注册的张数、发布与否、配准等级都相同，5 个作业的点数差 1–19 个点。

不开 (d) 时收下的照片和 D31 的运行逐栋相同（460 张）。**打开 (d) 的那次对它（A = `d34-dev-doff`，B = `d34-dev`）**：

| 守卫规则 | A | B | `compare` 的结果 | 说明 |
|---|---|---|---|---|
| `wrong_face_models_published` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `style_registration_wrong` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `style_qa_failed` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `struct_incorrect` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `struct_qa_failed` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `stale_overwrites` | 0 | 0 | 通过 | 0 on 5 building(s) |
| `models_published` | 66 | 66 | 通过 | same, delta +0 (threshold 2) |
| `facades_published` | 19 | 18 | 不过 | significantly worse: aggregate；汇总变化 -1，`compare` 的门槛 0，各栋记录的极差之和 2；按 dev 的噪声余地不算在 (d) 头上 |
| `wrong_face_cameras_shown` | 0.5 | 1 | 通过 | worse, delta +0.5 (threshold 6) |
| `gt_gross_shown` | 1 | 3.5 | 不过 | significantly worse: aggregate, brandenburg-gate, frauenkirche-dresden；汇总变化 +2.5，`compare` 的门槛 1，各栋记录的极差之和 1；**算在 (d) 头上** |
| `photo_pose_determined` | 310 | 308.5 | 通过 | worse, delta -1.5 (threshold 13) |
| `photo_pose_wrong` | 0.3186 | 0.2929 | 通过 | pooled 68.5/215 = 0.319 -> 61.5/210 = 0.293 (95% end -0.112); nearest building: leaning-tower-pisa 11.5/71.5 = 0.161 -> 13/71 = 0.183 (95% end -0.103) |
| `geo_scale_error_median_pct` | 12.4887 | 12.5688 | 通过 | worse, delta +0.0801 (threshold 3.0875) |
| `geo_yaw_error_median_deg` | 4.485 | 3.9413 | 通过 | better, delta -0.5437 (threshold 0.5) |

| 核对 | 结果 | 数 |
|---|---|---|
| `runs_comparable` | 通过 | 5 buildings x 2 repeats ran to the end with A's request counts, inputs identical, commit e3928f0 clean, config d34-heading-warn (d_on), code_sha256 e4fc7a57bf40 |
| `accepted_photos_rise` | 通过 | split 460 -> 472 (delta +12, threshold 0); every funnel equals the corrected prediction: True; florence-baptistery 51 -> 54, aachen-cathedral 75 -> 75, brandenburg-gate 136 -> 139, leaning-tower-pisa 115 -> 116, frauenkirche-dresden 83 -> 88 |
| `shown_pose_coverage` | 通过 | 0.8698 -> 0.8857 (delta +0.0159, threshold 0.1049, noise 0.1049) |
| `georeg_errors` | 通过 | checked models 85 -> 94; scale_rms_ln 0.379 -> 0.375 (margin 0.047: 5 % of A 0.01895, A r1/r2 0.004, B r1/r2 0.047) ok; yaw_rms_deg 9.55 -> 8.93 deg (margin 0.4775: 5 % of A 0.4775, A r1/r2 0.24, B r1/r2 0.16) ok; scale_off 19/85 = 0.224 -> 20/94 = 0.213 (95% end -0.133) ok; yaw_off 12/85 = 0.141 -> 14/94 = 0.149 (95% end -0.099) ok |
| `newly_accepted_photos` | 通过 | 朝错面那一半 PASS；比例那一半 not judged（它们按 SfM 显示、有参考的 6 个张次，错 0 个；别的照片 388 个里错 111 个） |

算在 (d) 头上的守卫：`gt_gross_shown`；按噪声余地放过的：`facades_published`；这一对不通过。

`gt_gross_shown` 对不开 (d) 的全新运行同样不过：汇总 1 → 3.5，勃兰登堡门 1 → 2.5、圣母教堂 0 → 1。不开 (d) 的全新运行在这两栋上和 D31 的运行一样（1 和 0，两次重复都是），所以先前那一对不过不是 SfM 重跑一遍或者 (a) + (b) 造成的，是 (d) 造成的。`facades_published` 19 → 18 在噪声余地之内。

**结论：(d) 不提名。** 两对都有一条守卫算在它头上，按冻结的规则只有这一个结果。留出集用默认配置（`eval/configs/default.json`，(d) 关）跑，只当作 (a) + (b) 的全新确认；`facing_away_warns_other_refs` 保持默认关，代码、开关和它的测试留着。

**新收下的 12 张照片做了什么**（这是报告，不是判据；判据是上面的守卫）。按栋合计，两次重复除了 SfM 位姿的误差以外都相同。逐张的记录不进仓库。

| 建筑 | 新收下 | 注册进任何模型 | 注册进已发布的模型（在里面没被标记） | 按 SfM 显示 | 只按位置显示 | 有参考位姿 | 其中在参考里朝着建筑（偏离中心方向 ≤ 60°） | SfM 位姿对 / 错（两次重复的张次） |
|---|---|---|---|---|---|---|---|---|
| 洗礼堂 | 3 | 3 | 3 | 3 | 0 | 0 | 0 | 0 / 0 |
| 亚琛 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 |
| 勃兰登堡门 | 3 | 3 | 3 | 3 | 0 | 2 | 2 | 4 / 0 |
| 比萨斜塔 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 0 / 0 |
| 圣母教堂 | 5 | 4 | 1 | 1 | 4 | 3 | 3 | 2 / 0 |
| **合计（每次运行）** | 12 | 11 | 7 | 7 | 5 | 6 | 6 | 6 / 0 |

- 有参考位姿的 6 张，在参考里**都是朝着建筑拍的**：朝向偏离"相机指向建筑中心"的方向 0.7°–32.9°，都在校验用的 60° 以内。它们原来被 `facing_away` 拒掉，3 张是因为朝向读数错了（和参考的朝向差 80°、114°、137°），3 张是因为 GPS 位置错了（离参考位置 24 m、48 m、102 m，朝向读数只差 6°、14°、33°：从错的位置看过去，对的朝向也成了"背对"）。评审说的"被拒的磁北照片里有不少其实拍的是这栋楼"，在这 6 张上都成立。
- 12 张里 7 张注册进了已发布的模型，在里面都没有被标记，按 SfM 的位姿显示；其中有参考的 3 张都对（两次重复共 6 个张次，错 0 个；不到 10 个，事先写好这一半只报告、不判）。没有一张落在朝错面的模型里。
- 另外 5 张只按位置显示（地标页上只有一个点，没有朝向），都不在已发布的模型里：
  - 比萨一张：只注册进一个没发布的整栋模型，它的 GPS 位置离参考 24 m。
  - 圣母教堂四张，逐张看过照片：
    - 一张拍的是一座雕像和后面另一栋楼，画面里没有教堂，哪个模型都没注册进去。这一张是真的没在拍这栋楼；(d) 打开以后这样的照片也会被收下，在地标页上显示成一个位置点。
    - 一张拍的是广场，教堂在画面右边三分之一；它的 GPS 落在教堂轮廓里面，离参考位置 48 m。它注册进了 3 个模型，都没发布，其中一个是北立面的（9 张注册 7 张）：在里面它被标成 `far_from_gps`（离自己的 GPS 115 m），这个模型 7 台相机有 2 台被标记，质量门没让它发布。D31 的运行和不开 (d) 的全新运行里，北立面的模型（8 张注册 6 张）是发布着的，但按参考它的尺度只有 0.31–0.32（小了三倍），显示的 4 个位姿全错。
    - 一张拍的是旁边的一座宫殿，教堂在画面边缘，GPS 差 102 m，只注册进两个没发布的模型。
    - 一张是教堂后殿的仰拍，注册进 7 个模型，都没发布。

**粗差相机为什么变多**（`gt_gross_shown`：正在显示的已发布模型里，没被标记、离参考位姿超过 20 m 或 45° 的相机；逐台看过）。多出来的没有一台是新收下的照片：

- 勃兰登堡门：多出来的都在东立面显示的那个模型里。D31 和不开 (d) 的全新运行，显示的都是第 101 个请求的作业建的模型（59 张注册 52 张，按参考尺度 1.01–1.05）。打开 (d) 以后东面多了 3 张照片，重建的时机跟着变，显示的是第 99 个请求的作业建的模型（61 张注册 53 张），两次重复的尺度是 1.09 和 0.99。这个模型里离参考 21–23 m 的相机第一次重复有 2 台、第二次有 1 台，刚过 20 m 的线，都是原来就收下的照片。同一批上传，两次重复自己就差 1 台。
- 圣母教堂：多出来的 1 台在东北立面的模型里（离参考 21 m）。打开 (d) 以后这个立面的作业在第 6 个请求就有 8 张照片可用（相邻扇区里新收下的照片也算进来），建出的模型 8 张注册 6 张，GPS 定不了尺度，等级 `prior_only`，之后没有再重建；不开 (d) 时显示的是第 11 个请求建的（8 张都注册，等级 `full`）。

所以这条守卫拦下的不是"新收下的照片位姿错了"，而是"多收几张照片以后，同一个立面换了一个模型，新模型里另外几台相机越过了 20 m 的线"。这是打开 (d) 的后果，规则把它算在 (d) 头上是对的；但它也说明两次重复估出来的噪声（同样的上传，SfM 的结果大部分逐个作业相同）盖不住"照片集合变了"这种变化。要单独判 (d)，需要它自己的一对运行，而且重复次数要够。

**逐栋并排**（核对 `side_by_side`，只报告）。每格依次是：D31 的运行（A）→ D31 的产物按 (a) + (b) 重放（A′）→ 全新，(d) 关 → 全新，(d) 开。都是每栋两次重复的中位数。

| 建筑 | 收下的照片 | 发布的模型 | 有模型的立面 | 显示的位姿 | 对 | 错 | 偏航误差中位 ° | 尺度误差中位 % | 墙离轮廓 > 1.5 m m | 地理位置差 > 5 m 的墙 m | 粗差相机 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 洗礼堂 | 51 → 51 → 51 → 54 | 6（都一样） | 5（都一样） | 39 → 39 → 39 → 42 | 11 → 12 → 12 → 12 | 6 → 5 → 5 → 5 | 5.135 → 4.805 → 4.775 → 4.83 | 10.985 → 10.775 → 10.765 → 10.7825 | 5（都一样） | 16.5 → 16.5 → 17 → 16 | 0（都一样） |
| 亚琛 | 75（都一样） | 3.5 → 3.5 → 3 → 3 | 2.5 → 2.5 → 2 → 2 | 41.5 → 41.5 → 40 → 40 | 1.5 → 1.5 → 1 → 1 | 0（都一样） | —（都一样） | —（都一样） | 0（都一样） | 0（都一样） | 0（都一样） |
| 勃兰登堡门 | 136 → 136 → 136 → 139 | 28（都一样） | 4（都一样） | 84 → 84 → 84 → 87 | 56.5 → 58.5 → 56.5 → 60 | 9.5 → 7.5 → 9.5 → 8 | 4.8725 → 4.255 → 3.98 → 3.8775 | 12.6425 → 12.5925 → 14.2125 → 14.4975 | 1 → 1 → 2 → 8 | 1 → 1 → 1 → 3.5 | 1 → 1 → 1 → 2.5 |
| 比萨斜塔 | 115 → 115 → 115 → 116 | 11 → 12 → 10.5 → 11 | 3.5 → 3.5 → 3 → 3 | 88 → 88 → 83.5 → 82.5 | 59 → 59 → 60 → 58 | 17 → 17 → 11.5 → 13 | 3.9525 → 4.225 → 4.195 → 4.005 | 5.0075 → 5.53 → 5.495 → 6.935 | 7 → 7.5 → 8.5 → 8.5 | 2 → 3 → 2.5 → 2.5 | 0.5 → 1 → 0 → 0 |
| 圣母教堂 | 83 → 83 → 83 → 88 | 20 → 19 → 18.5 → 18 | 5 → 5 → 5 → 4 | 64.5 → 64.5 → 63.5 → 57 | 14.5 → 18 → 17 → 17.5 | 46 → 42.5 → 42.5 → 35.5 | 7.23 → 6.46 → 6.2 → 3.87 | 17.0925 → 22.315 → 26.605 → 14.355 | 19.5 → 19.5 → 19.5 → 2 | 99.3 → 99.3 → 99.8 → 79.9 | 0 → 0 → 0 → 1 |
| **合计**（误差是各栋中位数的中位数） | 460 → 460 → 460 → 472 | 68.5 → 68.5 → 66 → 66 | 20 → 20 → 19 → 18 | 317 → 317 → 310 → 308.5 | 142.5 → 149 → 146.5 → 148.5 | 78.5 → 72 → 68.5 → 61.5 | 5.0038 → 4.53 → 4.485 → 3.9413 | 11.8137 → 11.6837 → 12.4887 → 12.5688 | 32.5 → 33 → 35 → 23.5 | 118.8 → 119.8 → 120.3 → 101.9 | 1.5 → 2 → 1 → 3.5 |

打开 (d) 的那一列里，圣母教堂离轮廓超过 1.5 m 的墙从 19.5 m 降到 2 m、错的位姿从 42.5 降到 35.5，是因为北立面那个尺度小了三倍的模型没有再发布（上面说的那张广场照片被标记以后，质量门把它拦下了），东北立面又换成了一个定不了尺度、不显示位姿的模型，不是 (d) 让哪个模型变准了；勃兰登堡门离轮廓超过 1.5 m 的墙从 2 m 升到 8 m，是东立面换了模型。两栋的方向相反，原因是同一个：照片集合变了，立面换了模型。

#### dev 上顺带量到的：(a) + (b) 在全新的 SfM 产物上（只报告）

不开 (d) 的那次全新运行（`d34-dev-doff`）和 D31 的运行收下的照片逐张相同，所以两次运行的作业一一对应（同一个请求、同一种作业、同一个扇区、同样多的照片）。它和 D31 的运行之间只差两样：(a) + (b)，和 SfM 重跑了一遍。把这次运行保留的模型按老规则（`d34-regeo/off.json`）和按运行自己的参数各重放一遍（`d34-fresh/d34-fresh-doff-{plain,own,off}-dev.json`），两样就分得开。dev 上这些只报告，不决定什么；(a) + (b) 由留出集的 `verdict_georegistration` 判。

每格依次是：D31 的运行，老规则（A）→ D31 的产物按 (a) + (b) 重放（A′，6.13 节）→ 全新 (d) 关，按老规则重放 → 全新 (d) 关，现在的默认值。

| 建筑 | 发布的模型 | 有模型的立面 | 整栋的模型 | 显示的位姿 | 对 | 错 | 偏航误差中位 ° | 尺度误差中位 % | 墙离轮廓 > 1.5 m m | 地理位置差 > 5 m 的墙 m | 粗差相机 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 洗礼堂 | 6（都一样） | 5（都一样） | 0（都一样） | 39（都一样） | 11 → 12 → 11 → 12 | 6 → 5 → 6 → 5 | 5.135 → 4.805 → 5.105 → 4.775 | 10.985 → 10.775 → 10.975 → 10.765 | 5（都一样） | 16.5 → 16.5 → 17 → 17 | 0（都一样） |
| 亚琛 | 3.5 → 3.5 → 3 → 3 | 2.5 → 2.5 → 2 → 2 | 0（都一样） | 41.5 → 41.5 → 40 → 40 | 1.5 → 1.5 → 1 → 1 | 0（都一样） | —（都一样） | —（都一样） | 0（都一样） | 0（都一样） | 0（都一样） |
| 勃兰登堡门 | 28（都一样） | 4（都一样） | 0（都一样） | 84（都一样） | 56.5 → 58.5 → 54.5 → 56.5 | 9.5 → 7.5 → 11.5 → 9.5 | 4.8725 → 4.255 → 4.8725 → 3.98 | 12.6425 → 12.5925 → 14.255 → 14.2125 | 1 → 1 → 2 → 2 | 1（都一样） | 1（都一样） |
| 比萨斜塔 | 11 → 12 → 9.5 → 10.5 | 3.5 → 3.5 → 3 → 3 | 0.5 → 0.5 → 0 → 0 | 88 → 88 → 83.5 → 83.5 | 59 → 59 → 60 → 60 | 17 → 17 → 11.5 → 11.5 | 3.9525 → 4.225 → 4.0275 → 4.195 | 5.0075 → 5.53 → 4.76 → 5.495 | 7 → 7.5 → 8.5 → 8.5 | 2 → 3 → 1.5 → 2.5 | 0.5 → 1 → 0 → 0 |
| 圣母教堂 | 20 → 19 → 20 → 18.5 | 5（都一样） | 0（都一样） | 64.5 → 64.5 → 63.5 → 63.5 | 14.5 → 18 → 13.5 → 17 | 46 → 42.5 → 46 → 42.5 | 7.23 → 6.46 → 7.51 → 6.2 | 17.0925 → 22.315 → 19.5325 → 26.605 | 19.5（都一样） | 99.3 → 99.3 → 99.8 → 99.8 | 0（都一样） |
| **合计**（误差是各栋中位数的中位数） | 68.5 → 68.5 → 66.5 → 66 | 20 → 20 → 19 → 19 | 0.5 → 0.5 → 0 → 0 | 317 → 317 → 310 → 310 | 142.5 → 149 → 140 → 146.5 | 78.5 → 72 → 75 → 68.5 | 5.0038 → 4.53 → 4.9887 → 4.485 | 11.8137 → 11.6837 → 12.615 → 12.4887 | 32.5 → 33 → 35 → 35 | 118.8 → 119.8 → 119.3 → 120.3 | 1.5 → 2 → 1 → 1 |

- **SfM 重跑一遍带来的差别**（第 1 格对第 3 格，都是老规则）：发布的模型 68.5 → 66.5。亚琛少 0.5 个：东南立面第 29 个请求的作业，全新运行两次重复都只注册了 24 张里的 5 张，D31 的运行有一次重复注册了 10 张。比萨少 1.5 个（D31 两次重复是 12、10，这里是 10、9）：第一次重复里整栋的模型和东北立面第 33 个请求的模型没有发布，第二次重复里西立面第 22 个请求的作业只注册了 25 张里的 4 张；整栋的模型全新运行哪次重复都没有，D31 的运行两次里有一次。显示的位姿 317 → 310，对的 142.5 → 140，错的 78.5 → 75。
- **(a) + (b) 在同一批全新产物上的效果**（第 3 格对第 4 格）：对的位姿 140 → 146.5，错的 75 → 68.5，和在 D31 的产物上重放出来的一样（+6.5、−6.5）；发布的模型 66.5 → 66（比萨 +1；圣母教堂 −1.5：西南面的模型被质量门的罗盘检查否掉，两次重复各 3 个和 4 个，西面多发布 2 个，6.13 节的重放里已经看到过）；偏航差超过 15° 的模型 14 个 → 12 个。按 `d34-v1` 的算法，这一对在 dev 上"有用"三条满足、"没坏处"每条满足、风格化不变差、两条规则通过；按运行自己的参数重放和不重放的补算逐值相同（239 个模型都重放了，没有一个变）。打开 (d) 的那次运行自己的配对重放（`d34-fresh/d34-fresh-{plain,own,off}-dev.json`，241 个模型）结论相同："有用"满足两条（对 +4.5、错 −4.5），别的都通过。

**作业逐个对**（`d34-dev-doff` 对 D31 的运行，两次重复合计）：

| 建筑 | 键相同的作业（两次重复合计） | 注册张数相同 | 注册张数和点数都相同 | 发布与否不同 | 配准等级不同 |
|---|---|---|---|---|---|
| 洗礼堂 | 14 | 14 | 8 | 0 | 0 |
| 亚琛 | 24 | 23 | 19 | 1 | 1 |
| 勃兰登堡门 | 90 | 89 | 67 | 0 | 3 |
| 比萨斜塔 | 50 | 34 | 15 | 5 | 1 |
| 圣母教堂 | 60 | 59 | 30 | 11 | 4 |
| **合计** | 238 | 219 | 139 | 17 | 9 |

238 个作业里 219 个注册的张数和 D31 的运行相同，139 个连点数都相同（留下的是同一个 SfM 模型）。注册张数不同的 19 个里 16 个在比萨。发布与否不同的 17 个里，11 个在圣母教堂，注册张数都没变，是 (b) 改了质量门的结论（重放看得到的那一部分）。

**重放看不到的部分：选候选和重试**。挑了 9 个作业，各从它自己的照片重新跑一遍 COLMAP（一次按现在的默认值，一次按老规则 `params.AS_BEFORE`，共 18 次），每次建图尝试出来的每个模型都按两套参数各配准、检查一遍，看两套参数各把哪个模型排在第一（`_Candidate.key`）。脚本在开发机的草稿目录里，不在仓库里。

| 作业 | 为什么挑它 | 重跑看到的 |
|---|---|---|
| 勃兰登堡门，东立面，第 44 个请求，25 张（取自打开 (d) 的运行，那时还没有新收下的照片） | 那次运行的第一次重复只注册了 13 张，别的运行、别的重复都是 22 张 | 两次重跑、每个种子都出 22 张的模型，两套参数排第一的是同一个。13 张那一次是 SfM 的偶然结果，不是排序造成的 |
| 勃兰登堡门，东南立面，第 69 个请求，9 张 | 等级从 `full` 变成 `prior_only` | 两套参数留下的是同一个 5 张的模型。老规则认为 GPS 定得了尺度（跨度 9.9 m），(a) 在拟合出的布局上量是 8.4 m，不认。重放看得到；这个模型两种情况下都没发布 |
| 亚琛，东南立面，第 29 个请求，24 张 | 全新运行注册 5 张，D31 有一次 10 张 | 两次重跑都只出 5 张的模型（591 个点），两套参数排第一的是同一个。10 张那一次没有重现 |
| 比萨，东北立面，第 33 个请求，9 张 | 注册张数相同，发布与否不同 | 同一个 7 张的模型，被标记的相机有时 1 台、有时 2 台，两套参数数出来的一样。差在 SfM，不在规则 |
| 比萨，西立面，第 22 个请求，25 张 | 全新运行有一次只注册 4 张，D31 是 8 张 | 两次重跑都出 8 张的模型，两套参数排第一的是同一个 |
| 比萨，整栋，第 33 个请求，109 张 | D31 有一次发布了整栋的模型（87 张），全新运行没有 | 每次都出两个模型：87 张的（19 台相机被标记）和把"双胞胎"图像对切掉重建的 73 或 75 张的（5 或 8 台被标记）。没被标记的相机是 68 对 68 或 67：打平时切过的那个因为罗盘误差小而胜出，差一台时 87 张的胜出，两套参数的结果相同。所以整栋模型有没有，是一台相机的事，和 (a) + (b) 无关 |
| 圣母教堂，东北立面，第 11 个请求，8 张（6 个磁北读数，没有真北读数） | 立面的等级 | 三个种子都出 8 张的模型。老规则挑罗盘误差最小的那个；现在磁北读数不算，三个打平，留下第一个。显示的东北面模型按参考尺度和 D31 的一样（0.58、0.62） |
| 圣母教堂，东北立面，第 6 个请求，8 张（取自打开 (d) 的运行） | 那次运行里这个立面是 `prior_only` | 6 张的模型，按现在的默认值 6 次尝试都定不了尺度，按老规则 6 次里 5 次也定不了。不是 (a) 单独造成的 |
| 圣母教堂，整栋，第 21 个请求，58 张（24 个磁北读数，1 个真北读数） | D31 留下 7 张的模型，全新运行留下 51 张的 | **这里两套参数不一样。** 老规则下有 8 台相机同时和 GPS、罗盘对不上，触发"切掉双胞胎图像对再建一次"，切出来的模型胜出（重跑里是 43 张，D31 的运行里是 7 张）。现在磁北读数不算，没有相机"同时和两样对不上"，不再切，留下 51 张的模型，其中 27 台相机离自己的 GPS 很远。两种情况下这个整栋模型都没有发布 |

18 次重跑里 14 次两套参数排第一的是同一个模型，亚琛、勃兰登堡门、比萨重跑的每个作业都是。不同的只在圣母教堂，两种：只有磁北读数时大小相同的候选打平；整栋作业不再因为磁北读数触发重建。**在这 9 个作业里，默认值下的选候选和重试没有让任何一个已发布或正在显示的 dev 模型变差**；丢掉的是磁北读数原来提供的"模型把两个立面混在一起了"这条线索。注册张数不同的 19 个作业里重跑了 4 个，其余 15 个（比萨 14 个、勃兰登堡门 1 个）没有重跑。

#### 留出集的结果：(a) + (b) 通过，(d) 不判

**只跑了一次。** 从带着 dev 结果和提名的提交 6213160 的干净工作树跑（2026-11-12 11:15–12:14，3,559 秒），配置是 `dev_outcome` 提名的 `eval/configs/default.json`（(d) 关）。6213160 相对 e3928f0 没有改 `crowd3d/`、`scripts/`、`eval/configs/` 的任何一个字节，所以这次运行的代码和两次 dev 运行逐字节相同（`code_sha256` 相同，有测试守着）。五栋、每栋两次重复都跑完，没有被时间上限截断；收下的照片和 D31 的运行逐栋相同（399 张）。结果 `eval/results/commons-ten-v1/d34-test.json`；之后的三条补算（不重放、按运行自己的参数重放、按老规则重放）从同一个提交跑，结果在 `d34-fresh/d34-fresh-{plain,own,off}-test.json`。看到结果以后没有调任何参数，留出集也没有再跑任何东西。

**`verdict_d`：不判。** (d) 在 dev 上没有被提名，留出集没有带着它跑，`facing_away_warns_other_refs` 保持默认关。

**`verdict_georegistration`：PASS。** 把这次运行保留的 145 个模型按老规则（对照）和按运行自己的参数（现在的默认值）各重放一遍，同一批 SfM 产物、同一批照片，没有运行间的噪声：

| `d34-v1` 的规则或核对 | 结果 | 数 |
|---|---|---|
| 按运行自己的参数重放，等于不重放的补算（前提） | PASS | 145 个模型都重放了，一个都没变（变换、等级、发布状态都是 0）；1590 个指标值、标定表 47 行都相同（另有 24 个只是用时和内存的值不同） |
| 有用（至少一条） | PASS | 满足三条：偏航均方根 7.69° → 6.84°（−11.1%）；错的位姿 35.5 → 32.5 而对的不减；对的位姿 61.5 → 64.5 而错的不增 |
| 没坏处（每一条） | PASS | 发布的模型 25.5 → 26.5；显示的位姿 186 → 186；超出自己 2σ 的 12.5 → 11.5；朝错面的模型和相机都是 0；四个 2σ 覆盖率不降（1、1、0.911 → 0.915、0.933 → 0.936）；尺度偏差超过 35% 的模型 6 → 6，偏航差超过 15° 的 2 → 1 |
| 风格化不变差 | PASS | 40 个值都不高于对照 |
| 朝错面的已发布模型为 0（`compare --criteria d34-v1 --noise-factor 0`） | PASS | 五栋都是 0 |
| 错位姿的比率不比对照差（同上） | PASS | 合并 35.5/97 = 0.366 → 32.5/97 = 0.335，每栋都不差（悉尼歌剧院没有参考位姿，不判） |

所以 `gps_span_model_side` 和 `compass_other` = `ignored` 保持默认，`yaw_arbitration` 保持关，`RECON_OUTPUT_VERSION` 还是 4（没有哪个默认值变了）。

**总体**

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh-off | 25.5 | 28 | 0.254 | 6/28 | 7.69 | 5.78 | 2/28 | 3.82 | 186 | 61.5 | 35.5 | 12.5 | 1 / 1 / 0.9111 / 0.9333 |
| fresh-own | 26.5 | 29 | 0.251 | 6/29 | 6.84 | 4.65 | 1/29 | 3.82 | 186 | 64.5 | 32.5 | 11.5 | 1 / 1 / 0.9149 / 0.9362 |

**配准等级，和重放改了什么**（两次重复都算）

| 组 | 模型 | 重放了 | 原样留下 | 变换变了 | 等级变了 | 发布状态变了 | full / prior_only / none | 已发布 |
|---|---|---|---|---|---|---|---|---|
| fresh-off | 145 | 145 | 0 | 96 | 2 | 2 | 94 / 51 / 0 | 51 |
| fresh-own | 145 | 145 | 0 | 0 | 0 | 0 | 96 / 49 / 0 | 53 |

**按模型注册的照片数分档**（每格：模型数，|ln 尺度| 均方根，尺度偏差 > 35% 的个数，偏航均方根 °）

| 组 | 3-6 张 | 7-10 张 | 11-20 张 | 21+ 张 |
|---|---|---|---|---|
| fresh-off | 0，—，0/0，— | 8，0.232，3/8，7.76 | 18，0.271，3/18，7.6 | 2，0.154，0/2，8.22 |
| fresh-own | 0，—，0/0，— | 9，0.22，3/9，7.2 | 18，0.274，3/18，6.48 | 2，0.154，0/2，8.22 |

**圣家堂**（`sagrada-familia`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh-off | 15 | 16 | 0.313 | 6/16 | 9.25 | 7.13 | 2/16 | 3.82 | 52 | 38.5 | 7.5 | 1.5 | 1 / 1 / 1 / 0.9 |
| fresh-own | 16 | 17 | 0.306 | 6/17 | 7.95 | 5.86 | 1/17 | 3.82 | 52 | 41.5 | 4.5 | 1.5 | 1 / 1 / 1 / 0.9062 |

**匈牙利国会大厦**（`hungarian-parliament`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh-off | 4.5 | 5 | 0.196 | 0/5 | 6.92 | 5.82 | 0/5 | 17.83 | 45 | 2 | 12 | 6 | 1 / 1 / 0.2 / 1 |
| fresh-own | 4.5 | 5 | 0.196 | 0/5 | 6.92 | 5.82 | 0/5 | 17.84 | 45 | 2 | 12 | 6 | 1 / 1 / 0.2 / 1 |

**林肯纪念堂**（`lincoln-memorial`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh-off | 2 | 2 | 0.123 | 0/2 | 3.81 | 3.8 | 0/2 | 8.85 | 31 | 5 | 15 | 4 | 1 / 1 / 1 / 1 |
| fresh-own | 2 | 2 | 0.124 | 0/2 | 3.43 | 3.42 | 0/2 | 8.85 | 31 | 5 | 15 | 3 | 1 / 1 / 1 / 1 |

**印度门**（`india-gate`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh-off | 4 | 5 | 0.052 | 0/5 | 1.99 | 2.22 | 0/5 | 2.75 | 48 | 16 | 1 | 1 | 1 / 1 / 1 / 1 |
| fresh-own | 4 | 5 | 0.052 | 0/5 | 1.98 | 2.22 | 0/5 | 2.75 | 48 | 16 | 1 | 1 | 1 / 1 / 1 / 1 |

**悉尼歌剧院**（`sydney-opera-house`）

| 组 | 发布的模型 | 能和参考比的模型 | \|ln 尺度\| 均方根 | 尺度偏差 > 35% | 偏航均方根 ° | 偏航中位 ° | 偏航 > 15° | 支点水平误差中位 m | 显示的位姿 | 对 | 错 | 超出自己的 2σ | 2σ 覆盖率：偏航 / 尺度 / 水平 / 倾斜 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fresh-off | 0 | 0 | — | 0/0 | — | — | 0/0 | — | 10 | — | — | 0 | — / — / — / — |
| fresh-own | 0 | 0 | — | 0/0 | — | — | 0/0 | — | 10 | — | — | 0 | — / — / — / — |

要照实说的：
- 和 6.13 节在 D31 的产物上重放的结果一样，好处几乎都在圣家堂一栋（偏航中位 7.13° → 4.76°，对的位姿 38.5 → 41.5、错的 7.5 → 4.5，多发布一个模型：南立面第 49 个请求那个注册了 10 张的模型从 `prior_only` 变成 `full`）。匈牙利国会大厦、印度门上规则看的数没有变化（发布状态和配准等级变了的模型，五栋里只有圣家堂那一个），林肯纪念堂只有偏航中位从 3.81° 变成 3.43°，悉尼歌剧院两边都没有模型。
- 尺度误差没有变（|ln 尺度| 均方根 0.254 → 0.251，偏差超过 35% 的还是 6 个）。评审第一条发现指望的"尺度误差变小"在全新的运行里也没有出现。
- 能和参考比的模型只有 28–29 个，"偏航差超过 15° 的模型少 2 个"那一条没有满足（2 → 1）。

**全新运行对 D31 的运行（只报告，不是判定）。** `d34b-v1` 的守卫规则是给 (d) 的结论用的，(d) 没有提名，这里不判。但这次运行和 D31 的运行之间只差 (a) + (b) 和 SfM 重跑一遍，照样列出来：

| d34b-v1 的守卫规则（这里只报告） | A：D31 的运行 | B：全新运行 | `compare` 的读数 | 说明 |
|---|---|---|---|---|
| `wrong_face_models_published` | 0 | 0 | 不变差 | 0 on 5 building(s) |
| `style_registration_wrong` | 0 | 0 | 不变差 | 0 on 5 building(s) |
| `style_qa_failed` | 0 | 0 | 不变差 | 0 on 5 building(s) |
| `struct_incorrect` | 0 | 0 | 不变差 | 0 on 4 building(s); left out without struct_models: sydney-opera-house |
| `struct_qa_failed` | 0 | 0 | 不变差 | 0 on 4 building(s); left out without struct_models: sydney-opera-house |
| `stale_overwrites` | 0 | 0 | 不变差 | 0 on 5 building(s) |
| `models_published` | 25.5 | 26.5 | 不变差 | better, delta +1 (threshold 1) |
| `facades_published` | 9 | 9 | 不变差 | same, delta +0 (threshold 0) |
| `wrong_face_cameras_shown` | 0 | 0 | 不变差 | same, delta +0 (threshold 6) |
| `gt_gross_shown` | 4.5 | 3.5 | 不变差 | better, delta -1 (threshold 1) |
| `photo_pose_determined` | 179.5 | 186 | 不变差 | better, delta +6.5 (threshold 13) |
| `photo_pose_wrong` | 0.3575 | 0.3351 | 不变差 | pooled 34.5/96.5 = 0.358 -> 32.5/97 = 0.335 (95% end -0.154); nearest building: hungarian-parliament 10/13.5 = 0.741 -> 12/14 = 0.857 (95% end -0.186); no rate: sydney-opera-house |
| `geo_scale_error_median_pct` | 12.1937 | 13.065 | 不变差 | worse, delta +0.8713 (threshold 3.0875) |
| `geo_yaw_error_median_deg` | 2.9875 | 4.0925 | 变差，超出噪声 | significantly worse: aggregate |

| 核对（这里只报告） | 读数 | 数 |
|---|---|---|
| `runs_comparable` | 成立 | 5 buildings x 2 repeats ran to the end with A's request counts, inputs identical, commit 6213160 clean, config default (d_off), code_sha256 e4fc7a57bf40 |
| `accepted_photos_rise` | 不判 | config d_off: (d) is off; split 399 -> 399 |
| `shown_pose_coverage` | 成立 | 0.8756 -> 0.8814 (delta +0.0058, threshold 0.1049, noise 0.1049) |
| `georeg_errors` | 成立 | checked models 28 -> 29; scale_rms_ln 0.248 -> 0.251 (margin 0.056: 5 % of A 0.0124, A r1/r2 0.007, B r1/r2 0.056) ok; yaw_rms_deg 7.26 -> 6.84 deg (margin 0.363: 5 % of A 0.363, A r1/r2 0.08, B r1/r2 0.3) ok; scale_off 5/28 = 0.179 -> 6/29 = 0.207 (95% end -0.180) ok; yaw_off 2/28 = 0.071 -> 1/29 = 0.034 (95% end -0.195) ok |

- 除了一条，没有哪条守卫变差到超出噪声：发布的模型 25.5 → 26.5，有模型的立面 9 → 9，显示的位姿 179.5 → 186，粗差相机 4.5 → 3.5，错位姿的比率 0.358 → 0.335；尺度误差中位 12.19% → 13.07%，在门槛（3.09）之内。
- **`geo_yaw_error_median_deg` 读出来是"变差，超出噪声"**（各栋偏航误差中位数的中位数 2.99° → 4.09°，门槛 0.5°）。逐栋看：圣家堂 5.90° → 4.76°，林肯纪念堂 3.69° → 3.43°，印度门 2.25° 不变，匈牙利国会大厦 2.29° → 6.60°。变差的只有匈牙利国会大厦。把这次全新运行按老规则重放，它也是 6.60°，整个划分的这个数是 5.20°，比现在的默认值（4.09°）还大，所以它不是 (a) + (b) 的配准算法造成的。这栋楼正在显示的西立面模型（第 41 个请求，38 张照片）在全新运行里是另一个 SfM 结果：两次重复注册了 29 张和 32 张，D31 的运行是 31 张和 29 张。这一栋 D31 自己两次重复的偏航中位就是 2.85° 和 1.72°，只有 2 个能和参考比的模型。这里原先写的是"它不是 (a) + (b) 造成的"，说过头了：重放看不到建图时选哪个候选。补查的结果在下面"评审之后的补记"。
- 作业逐个对：两次运行收下的照片相同，143 个作业的键相同（林肯纪念堂有 1 个只在 D31、2 个只在全新运行里），138 个注册的张数相同，94 个连点数都相同；注册张数不同的 5 个都在匈牙利国会大厦。

**评审之后的补记：匈牙利国会大厦这一栋的数**（标准文件 `held_out_outcome.addendum`；只报告）

没有跑 COLMAP。读的是两次运行的日志和摘要、提交的结果文件；另外把全新运行在这一栋留下的 30 个模型（每次重复 15 个），用作业给候选打分的同一段代码，按老规则和按默认值各判了一遍，不重新建图。请求号从 0 起（日志里印的多 1）。

这一栋的数：每次重复取能和参考比的模型（已发布、配准上了）的偏航误差绝对值的中位数，两次重复再取中位数。能比的全是西立面的模型：

| 西立面的作业 | D31 第 1 次 | D31 第 2 次 | 全新第 1 次 | 全新第 2 次 |
|---|---|---|---|---|
| 请求 25（15 张） | 没发布 | 没发布 | 没发布 | 6.97°（发布了） |
| 请求 28（18 张） | 0.40°（发布了） | 没发布 | 没发布 | 没发布 |
| 请求 32（21 张） | 5.30° | 0.94° | 4.65° | 5.82° |
| 请求 41（38 张，正在显示的模型） | 7.62°，不算（没有配准上） | 2.50° | 10.11° | 5.74° |
| **这次重复的中位数** | 2.85° | 1.72° | 7.38° | 5.82° |

38 张的那个作业（西立面唯一超过 30 张的：第一次尝试被标记的相机超过 5%，或者罗盘误差超过 20°，才再试一次）各次尝试：

| | 第一次尝试（注册 / 标记） | 第二次尝试（注册 / 标记） | 留下的 | 没被标记的相机：留下的对没留下的 |
|---|---|---|---|---|
| D31 第 1 次 | 31 / 6 | 21 / 1 | 第一次 | 25 对 20 |
| D31 第 2 次 | 31 / 7 | 29 / 2 | 第二次 | 27 对 24 |
| 全新第 1 次 | 31 / 8 | 29 / 3 | 第二次 | 26 对 23 |
| 全新第 2 次 | 32 / 6 | 29 / 6 | 第一次 | 26 对 23 |

- 整个划分的数是四栋的中位数。如果匈牙利国会大厦还是 D31 的 2.285°，整个划分会是 2.855°，比 D31 的 2.9875° 还小：读出来变差的就是这一栋。
- 逐栋看，`compare` 给这一栋的门槛是 1.56°（两次运行各自两次重复之间的差，取大的），变化 +4.3° 超出了它。这个噪声只是从两次重复、每次 2–3 个模型估出来的。
- 请求 32 四次都留下第一次尝试（注册 13 张、标记 2 台），偏航误差却是 5.30°、0.94°、4.65°、5.82°：同一个作业、同样的候选，D31 自己两次重复之间就差 4.4°。请求 28 和请求 25 各只在一次重复里发布。请求 41 在 D31 的两次重复里留下的就不是同一个候选；D31 第 1 次那个 31 张的模型差 7.62°，但没有配准上（31 个 GPS 位置里只有 15 个一致），没算进 D31 的数里。
- **排除了的**：
  - (a) + (b) 的配准算法：按老规则重放，这一栋 30 个模型没有一个换了等级或者发布状态，两次重复还是 7.38° 和 5.82°。
  - (b) 在建图时的作用：西立面 8 个作业用的是同样的 38 张上传，35 张没有罗盘读数，3 张标的是真北，没有 (b) 管得到的读数。留下的和没留下的候选，打分、重试、质量门的罗盘检查都不受 (b) 影响。
  - (a) 对留下的模型的打分：30 个模型按两套参数各判一遍，没被标记的相机数、被标记的是哪几台、配准上没有、等级、要不要重试，30 个都相同。
- **没排除的**：(a) 会不会让没被选中的候选排到前面。没留下的候选没有保存。全新运行两次重复里留下的候选都多 3 台没被标记的相机；要让老规则选出另一个，没留下的那个得少标记 3 台。要分开只能在留出集的照片上再跑 COLMAP，没有做。dev 上的旁证：本节 dev 部分那 9 个单个作业的重跑里，83 个候选按两套参数各判过，其中 30 个上 (b) 起不了作用，(a) 在这 30 个上没有改变任何一台相机的标记、配准上没有和等级。
- **(b) 在这栋楼丢掉的线索**（和 dev 上圣母教堂的一样）：整栋的作业（请求 31，47 张）在 D31 的运行里因为磁北读数触发过"切掉图像对再建"（第一次尝试罗盘误差 76.7°），另一次重复被质量门以"双胞胎""罗盘"拦下；全新运行里磁北读数不算，没有再建。全新运行留下的模型按老规则判，罗盘误差 67.2° 和 57.5°。请求 35 的整栋作业（58 张）在全新运行第一次重复里，按老规则会切了再建。这几个整栋的模型两种规则下都不发布。
- 这一条守卫读出来仍然是"变差"；结论和默认值不变（(a) + (b) 的结论只由配对重放决定）。

**逐栋并排**（核对 `side_by_side`，只报告）。每格依次是：D31 的运行，老规则（A）→ D31 的产物按 (a) + (b) 重放（A′，6.13 节）→ 全新，按老规则重放 → 全新，现在的默认值。

| 建筑 | 收下的照片 | 发布的模型 | 有模型的立面 | 显示的位姿 | 对 | 错 | 偏航误差中位 ° | 尺度误差中位 % | 墙离轮廓 > 1.5 m m | 地理位置差 > 5 m 的墙 m | 粗差相机 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 圣家堂 | 128（都一样） | 15 → 16 → 15 → 16 | 3（都一样） | 52（都一样） | 38 → 41 → 38.5 → 41.5 | 8 → 5 → 7.5 → 4.5 | 5.9 → 4.57 → 7.13 → 4.76 | 13.95 → 13.1175 → 13.475 → 12.96 | 15（都一样） | 65 → 64 → 66 → 65 | 0（都一样） |
| 匈牙利国会大厦 | 86（都一样） | 4.5（都一样） | 3（都一样） | 38.5 → 38.5 → 45 → 45 | 3.5 → 3.5 → 2 → 2 | 10 → 10 → 12 → 12 | 2.285 → 2.285 → 6.6 → 6.6 | 20.58 → 20.58 → 16.9025 → 16.9025 | 2.5 → 2.5 → 0.5 → 0.5 | 2.5 → 2.5 → 0.5 → 0.5 | 4 → 4 → 3.5 → 3.5 |
| 林肯纪念堂 | 58（都一样） | 2（都一样） | 1（都一样） | 31（都一样） | 4.5 → 4.5 → 5 → 5 | 15.5 → 15.5 → 15 → 15 | 3.69 → 3.355 → 3.805 → 3.425 | 10.4375 → 10.4725 → 13.125 → 13.17 | 1.5 → 1.5 → 3 → 3 | 1.5 → 1.5 → 3 → 3 | 0.5 → 0.5 → 0 → 0 |
| 印度门 | 65（都一样） | 4（都一样） | 2（都一样） | 48（都一样） | 16（都一样） | 1（都一样） | 2.25（都一样） | 5.22（都一样） | 19.9（都一样） | 0（都一样） | 0（都一样） |
| 悉尼歌剧院 | 62（都一样） | 0（都一样） | 0（都一样） | 10（都一样） | —（都一样） | —（都一样） | —（都一样） | —（都一样） | 0（都一样） | 0（都一样） | 0（都一样） |
| **合计**（误差是各栋中位数的中位数） | 399 → 399 → 399 → 399 | 25.5 → 26.5 → 25.5 → 26.5 | 9 → 9 → 9 → 9 | 179.5 → 179.5 → 186 → 186 | 62 → 65 → 61.5 → 64.5 | 34.5 → 31.5 → 35.5 → 32.5 | 2.9875 → 2.82 → 5.2025 → 4.0925 | 12.1937 → 11.795 → 13.3 → 13.065 | 38.9 → 38.9 → 38.4 → 38.4 | 69 → 68 → 69.5 → 68.5 | 4.5 → 4.5 → 3.5 → 3.5 |

仍然不行的楼，全新运行里和以前一样：悉尼歌剧院收下 62 张照片，一个模型都不发布；匈牙利国会大厦有参考的 14 个显示位姿里 12 个是错的，相机误差中位 29 m；林肯纪念堂 20 个里 15 个是错的。dev 上，亚琛收下 75 张只发布 3 个模型、只有 1 个对的位姿；圣母教堂显示的五个立面模型里四个按参考尺度差得远（0.31、0.58–0.62、3.2、2.4），有参考的 59.5 个显示位姿里 42.5 个是错的。这些都不是这一步的开关能解决的。

**显示范围的实测覆盖率，在这次运行上重算**（只报告；代码里的 `photoposes.MEASURED_COVERAGE` 引的还是 D31 的运行，这一步没有改它）：SfM 位姿 149/168 = 0.887（D31：146/165 = 0.885），单栋单次最低是匈牙利国会大厦第一次重复的 2/12、最高是圣家堂第一次重复的 44/44；手机 GPS + 罗盘 22/26（D31：23/28）。

**决定**（按冻结的标准，机械地）：
- `gps_span_model_side` = True、`compass_other` = `ignored` 保持默认：离线重放（6.13 节）和全新运行的配对重放在留出集上都通过。
- `yaw_arbitration` 保持关（不在这次要判的范围里）。
- `facing_away_warns_other_refs` 保持关：dev 上没有通过事先写下的守卫，没有提名，留出集上没有量过它。
- `RECON_OUTPUT_VERSION` 不变（4）：没有默认值变化，金样不动。

## 7. 每个指标是什么意思

全部指标定义在 `scripts/eval_real_landmark.py` 的 `METRICS` 里，排行榜末尾有英文原文。现行版本 `metric_schema_version` = 7。

- **方向**：↑ 越大越好，↓ 越小越好，"-" 只记录、不判好坏。
- **汇总**：跨建筑怎么合成一个数。sum = 相加；mean = 平均；median = 中位数；max = 取最大；pooled = 分子分母各自相加再相除（按模型合并，不是各建筑比率的平均）。
- 每个指标还有"最小变化量"和"相对下限"，比它们小的差值永远不算显著（例如运行时间差 5 s 以内或 25% 以内不算）。

### 7.1 输入和产品输出

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `photos_listed` | - | sum | 这栋建筑在测试集里、核对无误的照片数 |
| `photos_uploaded` | - | sum | 实际上传的照片数（自己的 EXIF 带 GPS 的；撞到时间上限时更少） |
| `photos_accepted` | - | sum | 通过产品校验的照片数 |
| `acceptance_rate` | - | mean | 通过数 / 上传数 |
| `sectors_with_photos` | - | sum | 至少有 1 张合格照片的方向数（共 8 个） |
| `sectors_face_ready` | - | sum | 照片够生成立面模型的方向数 |
| `jobs_run` | - | sum | 规划器启动的重建作业数 |
| `jobs_failed_sfm` | ↓ | sum | 重建直接失败（一个模型也没出）的作业数 |
| `models_published` | ↑ | sum | 通过质量门、发布了的模型数（含后来被替换的） |
| `models_qa_rejected` | ↓ | sum | 被质量门拦下、没发布的模型数 |
| `facades_published` | ↑ | sum | 最后有已发布立面模型的方向数 |
| `full_model_published` | ↑ | sum | 最后有没有发布全景模型（0/1） |
| `registration_rate` | ↑ | mean | 每个目标（每个立面、全景）最近一次尝试里注册上的照片比例；失败的作业算 0 |
| `estimate_coverage` | ↑ | mean | 地标页上显示了估计位置的合格照片比例 |
| `flagged_camera_fraction` | ↓ | mean | 最新发布的模型里被扣下不显示的相机比例（和自己的 GPS/罗盘矛盾、超出半径、焦距离谱） |

### 7.2 不用参考答案的位置检查

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `outline_rotation_abs_deg` | ↓ | median | 把每个最新发布模型的墙面点拟合到 OSM 建筑轮廓，需要转多少度。不依赖 GPS，检查北向对不对。对称建筑（八角形、圆形）上只是下限 |
| `outline_shift_m` | ↓ | median | 同样的拟合需要平移多少米。只拍到一面直墙时，只能看出垂直于墙的那部分 |
| `outline_residual_m` | ↓ | median | 拟合后墙面点离轮廓的中位距离：墙的形状对不对 |
| `outline_models` | - | sum | 墙面点够做这个拟合的模型数 |
| `gps_residual_median_m` | ↓ | median | 显示的相机位置离照片自己的 EXIF GPS 多远（部分循环：配准本身也用了 GPS） |
| `compass_error_median_deg` | ↓ | median | 显示的朝向和照片 EXIF 罗盘差多少度 |
| `compass_photos` | - | sum | 参与上一条的照片数 |

### 7.3 和参考答案比

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `gt_cameras` | - | sum | 最新发布的模型里有参考位姿的相机数 |
| `gt_rotation_error_median_deg` | ↓ | median | 模型整体对齐到参考后，每台相机的旋转误差中位数：模型形状对不对 |
| `gt_position_error_median_m` | ↓ | median | 同样对齐后的位置误差中位数（米） |
| `gt_gross_cameras` | ↓ | sum | 偏了 45° 以上或 20 m 以上的相机数（孪生面之类） |
| `gt_gross_shown` | ↓ | sum | 其中没被标记、会显示给用户的 |

**印度门参考的折叠（D31，参考版本 2，只改印度门）**：印度门的参考是一个 MegaDepth-X 分量，它把全部 41 台映射相机都放在拱门西侧：东西两面长得一样，东面的照片被配到了西面上（参考自己的 GPS 拟合 41 个里只有 19 个内点）。有 15 台参考相机离它们自己 EXIF 的 GPS 绕建筑差 124–179°（GPS 都在东侧）。逐张核对【目检加 EXIF，2026-11-03】：

- 7 张除了 GPS 还有别的证据说明参考错了：罗盘朝西（262°、269°、304°）；EXIF 时间是傍晚 18:11–18:35，按时间算太阳在 273–292°，照片里太阳或落日余晖正好在拱门里或门后（逆光），相机只能朝西；一张落日剪影；一张门后地平线上是黄昏余晖。
- 6 张只有 GPS 和参考相反，没有别的证据能定，作为"没证实"一并去掉。
- 2 张（"India Gate in Delhi (1)"、"(4)"）也和 GPS 相反，但参考是对的：拱门里能看见门东边的华盖（chhatri），所以是从西边拍的；这两张带的是同一个 GPS 标签（东北 242 m），是 GPS 错了。它们保持已验证。

所以第 2 版把这 13 台标成未验证（`eval_real_landmark.REFERENCE_FIXES`，`building_reference` 建参考时标，报告里记 `fix`）：照片位姿、地理检查、错面这些按"已验证参考相机"算的指标不再拿它们打分；`gt_model_check` 用的相对参考位姿不动（同一面的相机整体转了 180°，相互关系没变；`gt_*` 指标 `rescore` 也不重算）。其余建筑的参考版本和缓存键都不变（萨格拉达的参考模型已删，键一变就要重新下载 6.3 GB）。用现在的代码重建的第 1 版印度门参考和缓存逐位相同，所以第 2 版和第 1 版只差这 13 个标记。

效果（F 的 f-test 印度门两次运行，用 `rescore` 重算；F 没发布印度门的任何模型）：参考位姿 41 → 28 张，有定位的 6 → 3 张，对 3 → 3、错 3 → 0（原来"错"的 3 张都是带罗盘、按 GPS 和罗盘定位的东侧照片），`wrong_face_*` 0 → 0。按 D31 的立面规则把 F 的模型重新发布（探针 B 的做法），印度门东面两个模型（第 52、56 个请求，16/27、17/30）从"偏航 −178°、错面"变成没有已验证参考相机、无法核对；错面模型 2 → 0、错面相机 4 → 0，对 15 → 15、错 10 → 4（第 2 次运行 9 → 3），有参考的已定位照片 25 → 19（24 → 18）。基线和候选都要用同一版参考打分（6.11 节）。

### 7.4 运行时间

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `wall_time_s` | ↓ | sum | 这栋建筑整个运行的墙钟时间（不含建参考答案的时间：它是评测的输入，有缓存） |
| `jobs_time_s` | ↓ | sum | 重建作业时间之和 |
| `cpu_time_s` | ↓ | sum | 同一段时间里进程用掉的 CPU 时间（用户态 + 内核态，所有线程）。**主要的代价指标**：几个运行同时占着 4 个核时墙钟时间会变长，CPU 时间不会 |
| `jobs_cpu_s` | ↓ | sum | 重建作业的 CPU 时间之和 |
| `peak_rss_jobs_mb` | ↓ | max | 任何一次上传请求的作业期间进程的最高常驻内存：作业开始前把 VmHWM 复位（往 `/proc/self/clear_refs` 写 5），作业后读它；不让复位时退回整个进程的峰值 `ru_maxrss` |
| `baseline_rss_mb` | - | max | 同一次请求的作业开始前的常驻内存：峰值里不是作业加上去的那部分 |
| `request_latency_p50_s`、`_p90_s` | ↓ | median | 跑了作业的上传请求，从开始到作业做完的时间（中位数、90 分位）：上传的人要等多久模型才变 |
| `photo_time_to_pose_p50_s`、`_p90_s` | ↓ | median | 每张照片从它的上传请求开始，到地标页第一次显示它的位姿的时间（只算有位姿的照片；有多少张见 `estimate_coverage`） |
| `disk_run_mb` | ↓ | sum | 上传完后数据目录的大小（照片、模型、数据库；不含只有测试流程才保留的作业目录） |

机器同时在跑别的东西时墙钟时间、作业时间和延迟会被放大，比较时要看运行说明；CPU 时间和内存不受影响。但换了机器（例如容器重启后换了内核构建）CPU 时间也会变：6.11 节 F 和 B 差 0.85–0.90 倍，6.12 节 D31 和 D30 的运行差约 1.5 倍。所以跨运行比 CPU 时间，尤其是拿一次运行的 CPU 当另一次的分母时，要在同一台机器上带一个对照。补算（6.4）只能从请求记录里拿到延迟（第 7 版之后的运行还有作业 CPU 时间、峰值内存和到位姿的时间），`cpu_time_s` 和 `disk_run_mb` 照抄源结果。

### 7.5 风格化（`style_*`）

先说清楚三个词：

- **可参评**（eligible）：已发布、并且完成了完整地理配准（`georeg_level = full`）的模型。只靠罗盘和粗估尺度的 `prior_only` 模型照样风格化，但不计入通过率，而且不能用任何照片证据。
- **配准门**：吸附到 OSM 轮廓以后，墙点离轮廓足够近、逐边核验通过、并且没有任何歧义或倾斜等拒绝理由（docs/STYLIZED_MODEL.md §3.2）。没过门的模型不用照片证据：没有一面墙算"拍到"，照片量出的高度和立面起伏（D15、D24、D25）都不用，只有 OSM 和默认值的素模。过了门的模型同样是素模（D29 起不再上色），区别是照片证据可以用。
- **对 / 错**：过了门的模型，再拿轮廓校正后的参考位姿核对（至少 3 台相机有参考位姿才下结论）。四条都满足才算对：相机水平中心离参考的中位数 ≤ 1.5 m；还需要的转角 ≤ 5°；尺度差 ≤ 10%；墙点还要挪动的 90 分位 ≤ 1.5 m（取二维和三维里大的那个，倾斜也算）。

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `style_models_stylized` | - | sum | 跑了风格化的已发布模型数 |
| `style_models_eligible` | - | sum | 其中可参评的 |
| `style_registration_passed` | ↑ | sum | 可参评且过了配准门的 |
| `style_registration_pass_rate` | ↑ | pooled | 过门数 / 可参评数，跨建筑按模型合并 |
| `style_registration_checked` | - | sum | 过了门、又有 ≥ 3 台参考相机可以核对的 |
| `style_registration_right` | ↑ | sum | 其中对的 |
| `style_registration_wrong` | ↓ | sum | 其中错的：照片证据（D15、D24、D25 照片量出的数，照片位姿）会落到错的墙上或墙的错的位置。**这是最要紧的指标，目标是 0** |
| `style_rejected_placed_right` | ↓ | sum | 被门拒了、但单靠地理配准其实已经放对了的：本来可以用的照片证据没有用，是"宁可不用"的代价 |
| `style_gate_passed_not_eligible` | ↓ | sum | 不可参评却过了门的；必须为 0 |
| `style_snap_refused_gps` | - | sum | 吸附被拒，因为吸附后相机到墙的距离和照片自己的 GPS 对不上 |
| `style_snap_refused_ambiguous` | - | sum | 吸附被拒，因为另一个位置拟合得差不多（平行的面、隔一跨的位置、转一点或缩放一点也一样好） |
| `style_snap_residual_before_m` | ↓ | median | 吸附前（只靠地理配准）墙点离轮廓的中位距离 |
| `style_snap_residual_m` | ↓ | median | 吸附通过后的同一距离（门槛 1.5 m） |
| `style_ref_camera_error_m` | ↓ | median | 过门且可核对的模型，吸附后相机离参考位置的中位误差 |
| `style_ref_camera_median_worst_m` | ↓ | max | 上一条里最差的那个模型（超过 1.5 m 就是错） |
| `style_ref_camera_error_max_m` | ↓ | max | 最差的单台相机（单台相机自己的位姿错时，模型可能仍是对的） |
| `style_ref_walls_p90_worst_m` | ↓ | max | 墙点还要挪动的 90 分位里最大的那个 |
| `style_incomplete` | ↓ | sum | 风格化时间预算用完、没做到的模型数 |
| `style_walls_total` | - | sum | 每栋建筑"最佳模型"的墙数：最新发布的模型里过了配准门的优先，其次是完整模型，再次是注册照片最多的 |
| `style_qa_failed` | ↓ | sum | 没通过模型 QA 的模型数（风格化摘要的 `qa.passed`：诚实检查、`audit_model`、`audit_glb`、结构检查，并且没有无依据的结构数值）；**必须为 0** |
| `style_glb_kb_gz` | ↓ | max | 最佳模型 GLB 压缩后的大小 |
| `style_time_s` | ↓ | sum | 风格化总耗时 |

### 7.6 指标版本的变化

`compare` 不比较不同版本的结果，旧结果要在新版本下重跑一次基线。

| 版本 | 变化 |
|---|---|
| 1 | 第一版：输入、产品输出、轮廓、传感器、参考答案、运行时间 |
| 2 | 加 `style_*` 一组（风格化和配准门） |
| 3 | 加 `style_registration_right`、`style_rejected_placed_right`、`style_snap_refused_gps`、`style_snap_refused_ambiguous`；"对"= 转角 ≤ 5°、墙点中位处 ≤ 3 m、尺度 ≤ 15% |
| 4 | "对"收紧为转角 ≤ 5°、墙点 90 分位 ≤ 1.5 m、尺度 ≤ 10%；加 `style_gate_passed_not_eligible` |
| 5 | "对"还要求相机水平中心离参考 ≤ 1.5 m（中位数）、墙点按三维量（倾斜也算）；`style_ref_camera_error_worst_m` 改名 `style_ref_camera_median_worst_m`；加 `style_ref_camera_error_max_m`、`style_ref_walls_p90_worst_m`、`style_incomplete` |
| 6 | 加 `photo_pose_*` 一组（第 7.7 节）；之前的指标定义不变。测试流程版本 6：轮廓校正参考每栋建筑只算一次，风格化检查和照片位姿共用 |
| 7 | 加地理误差 `geo_*` 和 `wrong_face_*`、覆盖 `cov_*`（建筑局部和地理两个坐标系，1.5 m 和 5 m）、标定 `calib_*`、版本 `stale_overwrites` 和 `version_drift_*`（第 7.8 节）、运行代价 `cpu_time_s` 等（第 7.4 节）、`style_coloured_walls_wrong`（7.5 节；在还没有提交任何第 7 版结果之前加进第 7 版）、`cov_local_majority_m_1p5`、`cov_local_wrong_sum_m_1p5`、`cov_geo_wrong_sum_m_5`（7.8 节；B0 开始记录之后加进第 7 版：之前记的第 7 版结果里它们是空的，`rescore` 从产物补算，`compare` 记为"出现"），以及 `struct_*`（先占位，结构接进基准时原地填上，没升版本：`compare` 把"空 → 有值"记为"出现"，不算变差）；`calib_xy_*`、`calib_tilt_*` 在地理锚点那一步原地填上，有锚点的模型偏航和尺度也改用锚点的 σ（7.8 节；锚点之前的运行照旧，`rescore --regeo` 补出锚点）；之前的指标定义不变。测试流程版本 7：每次上传请求记下作业的 CPU 时间和峰值内存、之后地标页显示的位姿。**D29**（2026-11-03，原地，仍是第 7 版）：放弃上色，七个上色指标删掉（见表下） |
| 8 | D30：`photo_pose_*` 改为给别人看到的那一行打分（只算同意公开位置的照片，位置取整到 1 米、朝向取整到 1°，误差范围按取整放宽再向上取整），坐标系没定准的模型里的相机、磁北或没标注的罗盘不再显示（扣下）；加 `photo_pose_loc_*`（定位器没跑时为 0 或空）、`photo_pose_wrong_share`、`photo_pose_loc_wrong_face`、`photo_pose_withheld_right` / `_wrong`（第 7.7 节）；`photo_pose_loc_wrong`（`d30-v1` 的硬门槛）在还没有提交任何第 8 版结果之前加进第 8 版。别的指标定义不变，只把描述里"地标页显示的位姿"改成内部的相机估计（地标页已经不显示逐张照片的位姿）。D32（2026-11-05）不升版本：上传改成必须同意用户协议，公开的照片仍是每张通过的照片，打分的行不变 |

所以 v0 结果（版本 2）里的"错误通过"是按最松的规则判的；按版本 5 判，错的只会更多。

**D29：删掉上色指标，不升版本**（2026-11-03）。负责人决定放弃给模型上照片颜色，上色的代码要删掉，所以 `style_coloured_walls_wrong`、`style_walls_coloured`、`style_walls_unlocked`、`style_observed_area`、`style_roof_observed_area`、`style_fill_m2`、`style_provenance_violations` 这七个指标从 `METRICS` 里删掉了。指标版本仍是 7，理由：

- 剩下的指标在所有提交的输入上定义不变。这七个指标在提交的每个第 7 版 commons-ten-v1、mdx-two-v1 结果里都是 0（真实建筑上从来没有一面墙上过色）；只有 synthetic-v1 上过色（6 面墙）。
- 有两处定义变窄：上色的代码删掉以后，`style_qa_failed` 也就不再包括上色那几项检查；`style_walls_total`、`style_glb_kb_gz` 描述的"最佳模型"改为"过了配准门的优先，其次完整模型，再次注册照片最多"（原来是"上色墙最多，其次观测面积"）。在提交的每个第 5 版、第 7 版结果里，两种选法挑出的是同一个模型。
- 升到第 8 版的话，`compare` 要 `--force` 才能和提交的 B0、F 比（`--force` 连测试集和划分的检查也一起跳过），同一测试集下第 7 版的噪声块也都不能用了。
- 旧的结果文件里这七个键还在，没有代码再读它们：`compare`、`aggregate`、通过标准和排行榜都只按 `METRICS` 走，所以新结果和旧结果比时这七个指标根本不出现，不会记成"消失"。
- 每个模型的记录（`details.models[].style`）也变了：`walls` 只剩 `total`、`photographed`（算"拍到"的墙数，D24 的立面起伏只在这些墙上量）和两个平面拟合的最大偏差；`observed_pixel_area_fraction`、`roof_observed_area_fraction`、`fill`、`qa.honesty_passed` 和 `qa.violations` 删掉。

### 7.7 照片位姿（`photo_pose_*`）

**要回答的问题**：在三维模型上画出每张照片是在哪儿拍的、朝哪个方向拍的（"大致方位"），能画出来的有多少张、画得对不对。**没法确定方位的照片算失败。** 从指标第 8 版（D30）起，打分的是**别人看到的那一行**：只算上传者同意公开位置的照片（D32 起上传必须同意用户协议，就是每张通过、没撤回的照片），位置取整到 1 米、朝向取整到 1°，误差范围按取整放宽再向上取整（见下文"别人看到的那一行"）。粗糙的推断一律不显示：量不准的位姿被**扣下**（`withheld`），不会当成位姿画出来。

**每张照片用哪个来源**（`crowd3d/photoposes.py`；先有先用）：

1. `sfm`：照片被 SfM 注册进一个**已发布**、**坐标系定准了**的模型，相机没被标记。"定准了"指地理锚点的级别是 `full`（用 GPS 配准上了），或者风格化接受了轮廓吸附（吸附拿 OSM 墙核对过）；没有锚点的旧模型看 `georegistered`。同一张照片在多个模型里时取最新的，完整模型优先。模型的轮廓吸附被接受时，相机跟着吸附一起移动。俯仰角和横滚角由模型 JSON 里存的局部位姿（`cameras[pid].local`）经地理锚点算出；水平视角由相机的焦距比和处理后照片的宽高比算出。扣下的两种：
   - 坐标系没定准的模型（没能用 GPS 配准、也没有被接受的吸附）里的相机：偏航只来自罗盘或地标方向，尺度是估的（`_fallback_scale` 在真实建筑上偏大 1.5–2.4 倍）。mdx-two-v1 亚琛的这种模型里 4 台相机偏了 23.6–24.8 m。
   - 被标记的相机（离自己的 GPS 太远、和罗盘矛盾、焦距不正常）：F 里有参考位姿的 34 台这种相机对 17 台、错 17 台，没有独立的证据能把对的挑出来。
   - 照片在别的、坐标系定准了的旧模型里还有没被标记的相机时，显示那一台；更受偏好的模型里被扣下的那台照样记成扣下。
2. `loc`：视觉定位到一个已发布、坐标系定准了的模型里（产品的定位器是 `crowd3d/localize.py` 的 `ColmapLocalizer`，由 `jobs.JobRunner` 的定位作业跑，默认关（`Settings.localize_photos`；D30 第十一步按门槛判了，代价不过，保持关闭，6.12 节；关着时已存的结果也不显示，`jobs.shown_localizations`；基准给 `loc` 打分时用同一条规则，显示的就是打分的）：结果记录 {status, model_id, pose {enu, heading_deg, pitch_deg, roll_deg, hfov_deg, aspect}}，只用 `accepted` 的）。定位到坐标系没定准的模型里、或者照片自己的相机在已发布模型里被标记了，都扣下。
3. `exif`：照片自己 EXIF 里的 GPS 位置加罗盘方向，**只用标注为真北（`GPSImgDirectionRef` = `T`）的罗盘**，标作"手机 GPS + 罗盘（未经模型核对）"。高度没有测量，`p` 的第三个数为空；视角由 35 mm 等效焦距算，没有就为空。磁北或没标注的罗盘扣下（磁北标注常差约 90°，15° 内只有 47%，真北 85%）。
4. `gps`：有 GPS、没有可用的罗盘（没有罗盘，或者不是真北），SfM 和定位也没用上：只知道位置、不知道朝向，**算失败，也从不显示**。不会假设"照片一定朝着建筑"。
5. 被拒的上传：没有位姿，**算失败**。

**误差范围**（`sp` 米、`sy` 度，2σ；规则在 commons-ten-v1 和 mdx-two-v1 的运行上标定，位置和朝向各自约覆盖本来源 85–90% 的误差，样本内）。位置和朝向**同时**落在范围里的比例更低，而且误差有重尾（地理标签标错、罗盘差约 90° 或 180°），多大的圈都盖不全：`photo_pose_*uncertainty_coverage` 按来源报实测的覆盖率，结果的 `provenance.photo_pose_uncertainty_rule` 记着算它时用的 σ 规则和取整规则。标定用的照片包括 commons-ten-v1 没跑过 SfM 的五栋（它们当时按"仅 EXIF"打过分），所以 EXIF 的 σ 对这五栋不算样本外。第二轮修正给"没用 GPS 配准的模型"加过一行更宽的误差圈；D30 起这种模型的相机不再显示，那一行删掉了。

| 来源 | σ 位置 | σ 朝向 |
|---|---|---|
| `sfm` | max(15 m, 配准 RMSE, 相机离中心距离 × sin\|GPS 偏航 − 罗盘偏航\|) / 2 | max(√(偏航 σ² + 1), \|偏航冲突\| / 2, 10°)；配准自己报的偏航 σ（3–5°）不够，真实模型偏过 15–21° |
| `sfm`，吸附已接受 | ≤ 3 m | ≤ 3° |
| `loc` | hypot(该模型 SfM 相机在这一点的 σ, 1.5 m)：定位不会比模型自己的相机更准 | hypot(该模型 SfM 相机的 σ, 0.7°)（洗礼堂上相对误差中位 1.56 m / 0.51°） |
| `exif`（只有真北） | max(20 m, 手机报的 GPS 误差) | 15° |
| `gps`（不显示） | max(30 m, 手机报的 GPS 误差) | 无 |

EXIF 位置 σ 是第三轮修正改的（原来一律 12 m）：按 EXIF 画出来、有参考位姿的 51 张照片（commons，核心五栋的 r1 加仅 EXIF 五栋，每张只算一次）只有 37 张位置落在 24 m 的圈里、34 张位置和朝向都在。180 张带罗盘标注、有参考位姿的照片的原始 EXIF GPS（不管 SfM 有没有定位上）：`T` 标注的 90 分位 23 m，`M` 标注 60 m；真正按 EXIF 画出来的是 SfM 没定位上的那些，更差（`T` 的 85 分位 38–40 m）。σ 取 20 / 30 m 后，这 51 张里位置落在圈里的 43 张（84%）、位置和朝向都在的 40 张（78%）。剩下的是重尾：林肯纪念堂两张的 EXIF 位置离参考 109 m、353 m（参考相机在纪念堂里），圣家堂一张 182 m；印度门的参考有一部分被放到了拱门的另一面（第 7.3 节，参考第 2 版把这些相机标成未核实）。带罗盘标注又报了 `GPSHPositioningError` 的只有 mdx-two-v1 洗礼堂的 15 张（commons 的元数据里没有这个字段），14 张误差 ≤ 17 m，1 张 51 m（它报的是 5 m）：报的精度只用作下限，不用来缩小圈。

**别人看到的那一行**（`photoposes.public_row`，产品的公开接口和基准用同一个函数）：只有这几个键：`id`（HMAC(部署密钥, 照片 id) 的前 12 位十六进制）、`src`、`p`（东、北、高，四舍五入到整米；EXIF 的高为空）、`yaw`（整度，0–359）、`pitch`、`roll`（整度，没有就为空）、`hfov`（整度，D33 之前取到 5° 的整倍数；没有就为空）、`aspect`（两位小数）、`sp` = ⌈hypot(sp, 0.71 m)⌉、`sy` = ⌈sy + 0.5°⌉。0.71 m 和 0.5° 是取整最多能挪动的量，所以取整不会让误差范围变窄。行里没有上传者、时间、模型、扇区或照片 id。`public_poses` 只收公开的照片（`records.location_shared`：通过、没撤回），按 `id` 排序。基准每张照片都带着同意上传（测试流程版本 8）：D30 时记 `share-location-v1`，D32 起上传必须同意用户协议，记协议版本 `terms-v1`（`eval_real_landmark.UPLOAD_TERMS`，汇总里还是 `location_consent` 这个键）。测试流程版本没升：公开的照片还是每张通过的照片，分数不变，升版本只会让以前的运行键失效。`rescore` 把用户协议之前记录的运行（测试流程版本 < 8 没有同意记录，版本 8 在 D32 之前记的是 `share-location-v1`）的每张通过上传都当作同意（`eval_real_landmark.run_consent_assumed`），不依赖数据库打开时的 D32 迁移（补算只在临时副本上打开记录下的数据库），记在结果的 `provenance.location_consent` 里。D33 起用户协议在登记时同意：`run_crowd` 建好地标以后、计时的上传请求之前，每个伪用户经 `POST /api/contributors` 登记一次（同意 `UPLOAD_TERMS`，昵称是去掉首尾空白的伪用户名），再用这个登记上传；上传的照片和同意都和以前一样，没有指标读昵称，所以分数不变（测试里的合成众包场景用假重建器、用 COLMAP 加风格化各跑一次，改动前后的整份运行摘要，包括 143 个指标和逐张记录，除了时间、CPU、内存和随机 id 完全相同；COLMAP 那次限定一个建图线程（`IncrementalPipelineOptions.num_threads = 1`），因为多线程的 COLMAP 拿同一份代码跑两次就有光束法平差的噪声，一个线程时两次基线完全相同），测试流程版本仍是 8。

D33（2026-11-05）把视角的取整从 5° 改成 1°（`photoposes.PUBLIC_FOV_STEP_DEG`）。D33 之前记录的结果（`eval/results/` 里 D30 起的 11 个结果文件）的 `provenance.photo_pose_uncertainty_rule.public_fov_step_deg` 是 5，运行目录里每张照片记录的 `hfov` 是 5° 的整倍数。没有指标读 `hfov`（`eval_real_landmark.py` 第 2398 行只把它抄进记录），所以分数不受影响，结果文件不改，测试流程版本仍是 8。

**怎么算对**：有参考位姿（合成集用生成器真值；真实照片用第 7.3 节的轮廓校正 MegaDepth-X 参考，只用验证通过的分量）的照片，**别人看到的那一行**水平位置差 ≤ 10 m **且**朝向差 ≤ 15° 算对。10 m 能保证相机站在建筑的正确一侧、大致正确的位置（这些建筑宽 30–150 m）；15° 是产品自己的罗盘 σ，也不到手机视场角（约 60°）的一半，画出的视锥仍然对着同一片立面。另外报更严的 5 m / 10°。每张照片的记录（只在运行目录里，不进仓库）同时留着没取整的位姿（`p_exact`）和按它判的对错（`correct_exact` 等），汇总里的 `rounding` 数取整改变了几个判定。

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `photo_pose_determined` / `_rate` | ↑ | sum / pooled | 显示出位置和朝向的上传照片张数，和它占全部上传的比例（分母含被拒的） |
| `photo_pose_sfm`、`_loc`、`_exif` | - | sum | 各来源的张数 |
| `photo_pose_gps_only` | ↓ | sum | 有 GPS、没有可用的朝向：失败，不显示 |
| `photo_pose_gt`、`_gt_determined` | - / ↑ | sum | 有参考位姿的上传照片（含被拒的），其中确定了的 |
| `photo_pose_correct` / `_rate` | ↑ | sum / pooled | 有参考位姿的照片里，确定了**而且**对的；比率的分母是全部有参考的照片，没确定的和错的都算失败。**这是最诚实的通过率** |
| `photo_pose_wrong` | ↓ | sum | 确定了但错了 |
| `photo_pose_wrong_share` | ↓ | pooled | `wrong / gt_determined`：能核对的显示位姿里错的比例 |
| `photo_pose_correct_strict` / `_rate` | ↑ | sum / pooled | 按 5 m / 10° 算 |
| `photo_pose_pass` / `_rate` | ↑ | sum / pooled | 确定了、而且没被证明是错的 / 全部上传。没有参考位姿的照片无法核对，按通过算，所以偏乐观 |
| `photo_pose_testset_photos`、`_not_uploaded`、`_gt_testset` | - / ↓ / - | sum | 测试集里流程本该上传的照片（文件在、有 EXIF GPS；合成集全部）、其中被时间上限截掉没上传的、测试集里有参考位姿的（上传没上传都算） |
| `photo_pose_determined_rate_testset`、`photo_pose_correct_rate_testset` | ↑ | pooled | 同上两个比率，但分母换成整个测试集：没上传的照片也算失败。**每个测试集、每栋建筑的通过率按这两个报** |
| `photo_pose_pass_rate_low_testset` / `_high_testset` | ↑ | pooled | 全部测试集照片上的通过率下界（确定且对 / 全部：没参考的都算失败）和上界（确定且没被证明是错的 / 全部：没参考但确定了的算通过）。有参考位姿的照片偏向 SfM 定位到的那些，所以 `correct_rate` 要和这对上下界一起看 |
| `photo_pose_sfm_gt`、`_sfm_correct`、`_sfm_correct_rate`；`photo_pose_loc_…`；`photo_pose_exif_…` | - / ↑ | sum / pooled | 按来源分开的对错 |
| `photo_pose_position_error_median_m`、`photo_pose_heading_error_median_deg` | ↓ | median | 确定了的参考照片的误差中位数 |
| `photo_pose_within_uncertainty` / `photo_pose_uncertainty_coverage` | - | sum / pooled | 位置误差和朝向误差**同时**落在显示的误差范围（`sp`、`sy`）里的比例（每样各自约 85–90%，同时在就更低） |
| `photo_pose_sfm_within_uncertainty` / `photo_pose_sfm_uncertainty_coverage`；`photo_pose_loc_…`；`photo_pose_exif_…` | - | sum / pooled | 同上，按来源分开 |
| `photo_pose_loc_wrong` | ↓ | sum | 定位出的、有参考位姿、显示错的照片（`photo_pose_loc_gt - photo_pose_loc_correct`）。必须为 0（`d30-v1`） |
| `photo_pose_loc_wrong_face` | ↓ | sum | 定位出的位姿里朝向差 ≥ 30°、或者绕建筑转了 ≥ 45°（从地标中心看的方位角）的：看的是另一面墙。必须为 0 |
| `photo_pose_loc_queries`、`photo_pose_loc_cpu_s`、`photo_pose_loc_cpu_s_per_query` | - / ↓ / ↓ | sum / sum / pooled | 定位器试过的照片数和花的 CPU 时间（定位器没跑时为 0，每张的平均为空） |
| `photo_pose_withheld_right` | ↓ | sum | 有参考位姿、被扣下的位姿本来是对的照片（照片可能还有别的来源显示）：不显示粗糙位姿的代价 |
| `photo_pose_withheld_wrong` | - | sum | 有参考位姿、被扣下的位姿全是错的照片：这条规则挡掉的错误 |

结果里每栋建筑的 `poses.runs[]` 只有计数，没有逐张的数据：`withheld` 按"来源:原因"分对、错、没法核对，`withheld_right_not_shown_right` 是扣下的位姿本来对、现在又没显示对的照片数，`gt_by_source` 是各来源的对、严格对、在范围里、看错面的张数，`consent` 是通过的照片数、同意公开的张数和是否按旧库假设同意。跑了定位器（`rescore --localize`）的还有 `loc`：定位记录按状态（接受、拒绝、跳过）和原因数；显示出的定位位姿的张数和对、错、严格对、在范围里、看错面的张数；`gps_conflict`：其中定位器的 GPS 核对认为离照片自己的 GPS 太远的那些（`loc_with_gps_conflict`）的同样几个数；`accepted_not_shown`：接受了但没显示的（例如模型没发布）；`replaced`：每张定位出的照片原来显示的来源和"之前→之后"的对错（`right`、`wrong`、`not_shown` 有参考但没显示、`unchecked` 没有参考）；`by_class`：试过的照片里从没进过任何模型的、进过模型（没发布、坐标系没定准，或者没被显示）的各试了几张、接受了几张；`maps`：每张地图上显示了几张、对错、出了范围的张数和这张地图自己相机的误差中位数（第 7.8 节的地理核对），用来分开定位器的错和地图传下来的错；还有定位器自己的计数（试过的、要试的、候选地图数 `candidate_maps`、读进来的地图和没用的原因、图像对、碰到上限的、被拒的地图图像、地图特征 `map_features`）和 CPU（`query_cpu_s` 就是 `photo_pose_loc_cpu_s`）。

**在产品里**：地标详情已经不列逐张照片（D30 第一步）；公开位姿的接口 `GET /api/landmarks/{id}/poses`（D30 第四步）给出的就是这里的 `public_poses`（D32 起每一行的照片也能从 `.../poses/{id}/photo` 取到，基准不给照片打分），相机跟着风格化作业记下的、被接受的吸附移动，和基准的 `accepted_snaps` 同一条吸附规则（吸附接受了就算，不管配准检查过没过）。基准打分的行和别人看到的行不是按构造一致，可能有三处差别：基准在运行结束时给每张照片打分，接口只列最近一次重建作业结束前上传的照片（`as_of`）；基准用每个已发布模型的风格化记录里的吸附（记录状态 `ok` 就算，QA 没过也算），产品只用做出了 GLB 的模型的吸附（QA 通过、而且是这个目标最新发布的模型，被取代的模型不风格化）；D31 的运行里被接受的 4 个吸附都满足后者，所以没有出现差别。D30 验证的全新运行（6.12 节）在真实数据上核对了这两处：51 张照片都在快照里，接口给出的 40 行和基准规则（加上库里显示的定位结果）打出的公开字段逐行相同。视觉定位打开时还有第四处：`run` 在运行结束打分时不读库里的定位结果，`/poses` 却显示它们（6.12 节，要先接上才能打开定位）；定位关着（默认）时没有这处差别。另外，配准检查拒绝了的模型（GLB 标"素模（照片未对齐）"），只要吸附本身被接受，相机照样跟着吸附移动、误差范围照样封顶 3 m / 3°：D31 里只有合成集的 4 个模型是这样，吸附后的位姿都对（最大 1.22 m）；真实照片的模型还没有一个吸附被接受过，所以 3 m / 3° 的上限在真实照片上从没实测过。配准被"整个拒绝"（看不见的点、滑到对面、邻楼的墙）时要不要也不用吸附，D30 没有改：两边仍是同一条规则（吸附接受了就算）。

接口里每个来源附上的覆盖率（`photoposes.MEASURED_COVERAGE`）来自 commons-ten-v1 的测试集（有参考位姿的四栋：圣家堂、匈牙利国会大厦、林肯纪念堂、印度门），D31 全新运行按本节规则重算（基线 A，指标第 8 版）。两次运行合计：模型重建 146/165（4 栋），单栋单次最低 1/10（匈牙利国会大厦 r1）、最高 13/13（同一栋 r2）；EXIF 23/28（3 栋，其中 20 张是林肯纪念堂的），单栋单次最低 1/2（圣家堂）、最高 1/1。合计数会掩盖各栋之间的差别，所以接口和地标页同时给出最低和最高的一栋一次。**EXIF 这个数是样本内的**：EXIF 位置 σ 的标定照片包括这几栋（上面"误差范围"一段和第 2 节"已知的污染"），所以接口里记 `rule_calibrated_here: true`，地标页写明"不是独立检验，实际可能更低"；模型重建的 σ 规则是在这几栋还没跑过 SfM 时定的（`false`）。两者都不是整条流水线的留出检验：D31 的照片资格和立面过门阈值是看过这五栋定的（第 2 节"D31 的污染"）。D30 验证运行的 A（716de16，`eval/results/commons-ten-v1/d30-a-test.json`）逐张和这些数的来源相同，所以数不变。与模型比对（`loc`）在测试集上没有有参考位姿的定位位姿（B 只在印度门定位出 1 张，没有参考），所以仍写"还没有实测"；dev 上 11 个张次（6 张照片）都在范围里，是样本内的，而且产品里定位是关着的（6.12 节）。

### 7.8 地理误差、覆盖、标定、版本（第 7 版）

**要回答的问题**：模型放在地图上的位置本身错多少（不是对齐之后的形状误差，那是 7.3 节的 `gt_*`），它的不确定度说得准不准，轮廓上有多少米被模型真正支持，新版本有没有用旧照片盖掉新照片。

**地理误差**（`geo_check`，不做轮廓吸附）：已发布、完成地理配准、并且至少 3 台没被标记的相机有验证过的参考位姿的模型（被替换的也算）。相机误差 = 相机水平中心离参考位置的距离（每个模型取中位数）。有 ≥ 3 台相机知道朝向时（从保留的 `sparse/0` 读出旋转，按模型 JSON 的相机位置转到 ENU，和 `stylize_model.load_job` 一样），算把模型挪到参考上的三维相似变换（`correction_3d`：旋转来自相机朝向，所以倾斜也看得到），它的偏航、尺度、倾斜就是锚点的误差，它把墙点（离地 2–20 m）挪动的 90 分位是墙上的误差。不保留朝向的重建器只有相机误差。

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `geo_models_checked` | - | sum | 参与核对的模型数 |
| `geo_camera_error_median_m` | ↓ | median | 相机水平中心离参考的距离：地标页显示的位姿在地图上偏多少 |
| `geo_yaw_error_median_deg`、`geo_scale_error_median_pct`、`geo_tilt_error_median_deg` | ↓ | median | 锚点的偏航（绝对值）、尺度（\|s − 1\|，百分比）、倾斜 |
| `geo_wall_error_p90_median_m` | ↓ | median | 墙点要挪动的 90 分位（三维） |
| `wrong_face_models_published` | ↓ | sum | 偏航差 30° 以上的模型：朝向放反了（对称的错位、偏航失败） |
| `wrong_face_cameras_shown` | ↓ | sum | 最后地标页显示的位姿（每张照片一个，`camera_estimates`）里，朝向和参考差 30° 以上、或者站在建筑另一侧（从地标中心看方位差 45° 以上）的 |

**支持的覆盖**（`coverage_bins`，只看最后显示的模型，每个目标最新的那个）：OSM 外轮廓按弧长切成约 1 m 一段。墙点（离地 2–20 m）落到离它最近的轮廓位置所在的那一段，前提是离轮廓不超过 5 m；一个模型有 ≥ 10 个点落在某段上，就算覆盖了这一段。两个坐标系：

- **建筑局部**：先用模型到参考的相似变换把墙点挪过去（去掉地理配准的误差），误差 = 挪过去的点离轮廓的水平距离：墙的形状对不对。
- **地理**：误差 = 显示出来的点离"它真正所在的轮廓位置"（挪过去的点落到的轮廓点，高度取挪过去的点的高度）有多远：地理配准的误差也算在里面。

某个门槛下，一个模型在这一段上有 ≥ 10 个点的误差不超过门槛，这一段就算被支持（数点、不取中位数：柱廊后面、凹进去的地方看到的点不会把墙上的点压下去）。按负责人问题 1 的默认答复：**主要指标是建筑局部 1.5 m**（`cov_local_supported_m_1p5`），次要的是地理 5 m，地理 1.5 m 只作参考（现在预计接近 0：核心五栋的墙点地理误差中位约 5 m）。轮廓当作真实的墙面：OSM 本身有 0.5–1 m 的误差，柱廊、凸出的部分也离轮廓有一段，所以 1.5 m 的判定里带着这些（两边比较时是共同的）。仓库里没有 scipy，用参考点云当墙面要自己做最近邻，就没有用。

**数点随点的密度变大**：数点的规则只要一段上有 10 个点在门槛内，不管它们占落在这段上的点的多少，所以点云越密，被支持的米数越多，即使它一点也不更准：同一片点云密一倍，原来墙上只有 5 个点的段就有 10 个了（`tests/test_eval_real.py` 的覆盖测试）。所以并列报告按份额的 `cov_local_majority_m_1p5`：一个模型在这一段上建筑局部 1.5 m 内的点既要 ≥ 10 个，又要占落在这段上的点的至少一半（`COV_MAJORITY`）。它不会只因为点更密而变大，代价是柱廊后面看到的点占多数的段它不算。v1-after-fixes 核心五栋补算（0.2 的参考缓存，r1 / r2）：勃兰登堡门 1.5 m 支持 15.9 / 9.0 m，按份额只有 1.0 / 1.0 m；比萨 9.0 / 9.0 m 对 3.0 / 4.0 m；圣母教堂 79.9 / 76.9 m 对 74.9 / 75.9 m；洗礼堂 35.0 m 两边一样，合成集也一样。同一次补算的 r1，按数点被支持的段（每个模型各算）里，落在段上的点在 1.5 m 内的份额中位数：勃兰登堡门 0.19（16 段里 12 段不到 25%），比萨 0.40，圣母教堂 0.94，洗礼堂 1.00。持久地图让同一地标的点越积越多，和基线比的时候两个数要一起看：主要指标变好而份额指标没变，多半只是点更密了。

**每个显示的模型画错的墙**：两个覆盖守卫 `cov_local_wrong_m_1p5`、`cov_geo_wrong_m_5` 按段取并集：一段只要有一个显示的模型支持，别的显示的模型在这段上画错的墙就不算；一段被几个模型画错也只算一次。`cov_local_wrong_sum_m_1p5`、`cov_geo_wrong_sum_m_5` 把每个显示的模型自己覆盖却不支持的米数加起来（每个模型的数在 `details` 的 `coverage.local_wrong_1.5_m`、`geo_wrong_5_m`），和守卫并列报告，不是通过标准。同一次补算（r1）：勃兰登堡门建筑局部 44.8 m 对加起来 73.7 m、地理 27.9 对 56.7 m，洗礼堂地理 16.0 对 42.0 m，比萨建筑局部 7.0 对 16.0 m；合成集 synthetic-ring 地理 0 对 6.0 m（东南立面模型的墙在 5 m 外，完整模型支持同一段）。

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `cov_claimed_m` | - | sum | 显示出来的墙落在轮廓上的米数（按显示的位置，不需要参考） |
| `cov_unverified_m` | - | sum | 其中只有没法和参考对齐的模型（少于 3 台有朝向的参考相机）占着的米数 |
| `cov_local_supported_m_1p5` | ↑ | sum | 建筑局部 1.5 m 内支持的米数。**主要的覆盖指标** |
| `cov_local_supported_m_5` | ↑ | sum | 建筑局部 5 m：离轮廓超过 5 m 的点不算落在轮廓上，所以它就是被覆盖的全部米数 |
| `cov_geo_supported_m_1p5`、`cov_geo_supported_m_5` | ↑ | sum | 地理 1.5 m、5 m 内支持的米数 |
| `cov_local_wrong_m_1p5` | ↓ | sum | 被覆盖、但没有模型在建筑局部 1.5 m 内支持的米数（= 局部 5 m − 局部 1.5 m） |
| `cov_geo_wrong_m_5` | ↓ | sum | 被覆盖、但没有模型在地理 5 m 内支持的米数 |
| `cov_local_majority_m_1p5` | ↑ | sum | 一个模型在这段上建筑局部 1.5 m 内的点 ≥ 10 个、并且占落在这段上的点至少一半的米数：主要指标按份额的伴随数，不随点的密度变大 |
| `cov_local_wrong_sum_m_1p5`、`cov_geo_wrong_sum_m_5` | ↓ | sum | 每个显示的模型自己覆盖、却不在建筑局部 1.5 m（地理 5 m）内支持的米数，加起来：别的模型支持的段也算 |

**标定**（`calib_*`）：每个有三维相似变换的核对模型，每个分量算 z = 误差 / √(模型自己的 σ² + 参考的 σ²)。模型的 σ 来自它的地理锚点（`meta["anchor"]`，`crowd3d/anchor.py`），按现在 `Params` 的下限重新算（`GeoAnchor.recalibrated`：每个分量 σ = √(锚点自己的 σ² + 下限²)），所以补算不会把不同的标定混在一起；锚点出现之前的模型 JSON 只有偏航（`yaw_sigma_deg`）和尺度（`scale_sigma`），没有水平位置和倾斜（这两组为空，`rescore --regeo` 能补）。
- 偏航：\|偏航\|；尺度：\|ln s\|，都来自核对的相似变换。相似变换的尺度 s ≤ 0（退化的拟合，B0 里 mdx 亚琛 r1 有一个 -0.0002）没有尺度误差，这个模型不进尺度的标定，其余分量照算。
- 水平位置（`xy`）：锚点枢轴（GPS 内点相机的加权中心，`pivot_enu`）的东、北误差，模型减参考。
- 倾斜（`tilt`）：相似变换旋转第三列绕东、绕北两个轴的倾角。
- 参考的 σ：偏航 1°、尺度 2%（`sigma_ref`，缓存里存的）；水平位置每轴取参考已核验分量的墙面中位距离（贴轮廓的残差），至少 0.5 m；倾斜每轴 0.5°。合成集的生成真值都是 0。
- 水平位置和倾斜是两个轴、每轴一个 σ：z 取和这个二维半径（χ²，2 个自由度）尾部概率相同的正态 z，所以四个分量都是 1、2、3 个 σ 内应有 0.683、0.954、0.997。
- 真实数据上尺度的标定部分是循环的：参考的尺度本身也来自 Commons 照片的 GPS；高度（`up`）没法标定，参考用的是同一个相机高度假设。

每个核对模型的误差、σ（锚点自己的 σ 记在 `core`）、参考 σ 和 z 记在 `details.models[].geo.calib`，每次重复的全部记在建筑记录的 `calibration` 行里（`key` 是模型所有相机的位置：两次重复做出同一个模型时 key 相同）。`benchmark.py calibration` 读这些行出标定表（6.6 节）。

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `calib_<分量>_n` | - | sum | 有这个分量的误差和 σ 的模型数 |
| `calib_<分量>_within2` / `calib_<分量>_cover2` | - | sum / pooled | \|z\| ≤ 2 的个数和比例（σ 准的话约 0.954；少于 20 个模型说明不了什么） |
| `calib_<分量>_beyond3` | ↓ | sum | \|z\| > 3：地理配准说自己比实际准得多 |
| `calib_sigma_missing` | ↓ | sum | 没有锚点的核对模型里没有偏航或尺度 σ 的（吸附就没有窗口可搜；锚点总有每个 σ） |

**版本**（从数据库算：模型、它的作业用了哪些照片、照片的上传时间）：

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `stale_overwrites` | ↓ | sum | 替换了同一目标上一个已发布模型、而上一个模型有它没有的、上传得比它所有照片都晚的照片：旧的输入盖掉了新的。**必须为 0** |
| `version_drift_median_m`、`_p90_m` | ↓ | median | 同一目标相邻两个已发布版本，共同的未标记相机挪动的中位数；再在所有相邻版本对上取中位数、90 分位 |

**结构**（`struct_*`）：地标在 `eval/landmarks.json` 里有 OSM 分件快照（`osm_parts`，按 slug 对应）时，风格化这一步走 `crowd3d.structure_pipeline`，和 `scripts/stylize_model.py` 一样：分件就是体块；照片的配准被接受、分件不超过 60 个时，才用 A+R 天空掩膜、SfM 点和三维线判断（D15）；D25 只在有两种可商用的天空掩膜时才跑，所以现在不跑（G5 要两种掩膜）。没有快照的地标（合成测试集）这些指标为空。第 7 版里这些指标一直留着空位，现在原地填上，不升版本（`compare` 把"空 → 有值"算作"出现"）。每个模型的记录多一个 `structure` 块（判断是否跑了、原因、采纳、检查、时间、内存、是否在 1200 秒 / 4 GB 之内）。

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `struct_models` | - | sum | 用分件快照做的风格化模型数 |
| `struct_d15_adopted` | - | sum | 已发布模型里还生效的 D15 采纳（照片量出的高度和轮廓） |
| `struct_d25_records` | - | sum | 已发布模型里还生效的 D25 记录 |
| `struct_photo_share` | - | mean | 结构里的数字由照片量出的比例（`crowd3d.structmetrics`），模型间平均 |
| `struct_checked` | - | sum | 照片量出、已发布的高度和下限（D15、D25）里能和参考点云比的个数：参考答案已核实分量的点在该分件露出部分上的 0.98 分位高度（和 SfM 通道量顶的方法一样） |
| `struct_incorrect` | ↓ | sum | 其中超出 D15 一致容差（max(1 m, 5 %)）的个数；下限只要参考在容差内达到它就算对。**必须为 0** |
| `struct_unbacked` | ↓ | sum | 记录里说是照片量的、但证据对不上的结构数值。**必须为 0** |
| `struct_adopted_unregistered` | ↓ | sum | 配准门拒了的模型里还生效的 D15 采纳。**必须为 0** |
| `struct_qa_failed` | ↓ | sum | 结构质检（G1–G11 和无依据数值）没过的模型数。**必须为 0** |
| `struct_time_s` | ↓ | sum | 结构这一步的时间（读快照、掩膜、三维线、判断、D25，不含风格化），模型间求和 |
| `struct_peak_rss_mb` | ↓ | max | 一个模型的结构这一步连同它的风格化的峰值内存，取各模型最大 |

现在的众包模型配准都被拒（`style_registration_passed` 为 0），所以它们的判断都跳过：得到的是 OSM 分件的素模，照片量出的数字为 0。参考场景只用 A+R 时也是 0（V0）。这是现状，报告里照实写。

## 8. 噪声基线：多大的变化才算"真的变了"

**为什么需要**：COLMAP 在 CPU 上不是确定性的（多线程匹配和光束法平差），同样的输入跑两次，结果会不一样。照片少的建筑尤其明显，会在几种完全不同的结果之间跳。不知道波动有多大，就分不清"改进"和"运气"。

**怎么测**：同一配置用 `--repeat N` 跑 N 次，结果里的 `noise` 块记下每栋建筑每个指标的极差（最大值 − 最小值）、跨建筑的最大值、汇总值的极差，以及哪些指标时有时无。`compare` 自动拿它当门槛。

**现有的噪声记录**：

| 测试集 | 运行 | 指标版本 | 重复 | 建筑 | 值得注意的极差 |
|---|---|---|---|---|---|
| synthetic-v1 | `v0-noise` | 2 | 3 | 两栋 | 几乎为 0：轮廓转角 0.025°，GT 位置 0.004 m，运行时间 6.2 s |
| mdx-two-v1 | `default-noise` | 1 | 5 | 两栋 | 亚琛的轮廓转角 45.7°、GT 位置 12.7 m；亚琛有时发布 1 个模型、有时 0 个 |
| mdx-two-v1 | `v0-noise` | 2 | 5 | 两栋 | 亚琛的轮廓转角 26.9°、GT 位置 3.16 m |
| commons-ten-v1 | `v0-noise` | 2 | 3 | 洗礼堂、亚琛 | 洗礼堂每次通过配准门 1 / 3 / 2 个模型：`style_registration_passed` 极差 2，通过率 0.4；运行时间极差 593 s（有别的工作流同时占 CPU，被放大了） |
| commons-ten-v1 | `v1-after-fixes` | 5 | 2 | 核心五栋 | 发布模型数极差 1，注册率 0.071，轮廓转角 3.3°，运行时间 260 s；通过率两次都是 0 |
| mdx-two-v1、synthetic-v1 | `v1-after-fixes` | 5 | 2 | 两栋 | synthetic 极差接近 0 |
| mdx-two-v1、synthetic-v1 | `v1-merged` | 5 | 2 | 两栋 | 合并后的代码；synthetic 只有运行时间有极差（0.7 s）；mdx 洗礼堂轮廓转角 0.13°，亚琛两次都发布 1 个模型（`v1-after-fixes` 是一次 1 个、一次 0 个） |

**用的时候要注意**：

- **次数少会低估噪声**。早先 3 次重复的记录完全没碰到亚琛"一个模型都不发布"的结果；5 次重复之后，一个对这些照片什么都没改的配置（`fast` 在 1000 px 的照片上）仍被 `compare` 报出 3 个"显著变好"。照片少的建筑建议 ≥ 10 次，比较的两边都要重复。
- **只用同一指标版本的噪声**。版本 7 的噪声记录是基线 B0 那一组结果（都是两次重复）：synthetic-v1 和 mdx-two-v1 的 `b0` 以及 `v1-merged` 的第 7 版补算，核心五栋 `v1-after-fixes` 的第 7 版补算，留出五栋 `b0-test`；再加上冻结候选 F 的 `f`、`f-dev`、`f-test`，和 D31 候选的 `d31`、`d31-dev`、`d31-test`（6.11 节）。这些结果都在 `eval/results/` 里（6.10 节）；按参考第 2 版补算的 `commons-ten-v1/d31-baseline/f-test-ref2.json` 放在子目录里，不算噪声来源。旧运行用 `rescore` 补算到版本 7 时，各次重复分别重算，噪声块也跟着重算。版本 8（D30）的噪声记录是 D31 全新运行在 716de16 上的补算 `d30-a`、`d30-a-dev`、`d30-a-test`（6.12 节）；定位打开的 B 和那两次全新运行放在各测试集的 `d30-localize/` 子目录里，不算噪声来源。
- **dev 和 test 的结果在同一个目录里时**，`compare` 把目录里所有同版本结果的噪声块都算进门槛，汇总值的极差不按建筑分开，另一个划分的汇总极差也会混进来（例如 `b0-test` 对 `f-test` 的主要指标门槛从 2 m 变成 9.9 m，这是核心五栋 `v1-after-fixes-s7` 的汇总极差）。V5 的比较只用两个结果自己的噪声块；要复现，加 `--results-dir <空目录>`（[REVIEW_ADOPTION.md](REVIEW_ADOPTION.md) 第 7 节）。
- **commons-ten-v1 的版本 2 噪声只覆盖洗礼堂和亚琛**，其他建筑在那次比较里没有噪声估计（按 0 算），很小的差值也会被标成"显著"。
- **机器忙的时候运行时间不可比**。运行说明里要写清楚当时有没有别的负载。

## 9. 怎么切一个新的测试集版本

适用于：想加建筑、换照片、换轮廓或参考模型，或者发现清单有错。

1. **不要改已发布的测试集**。起一个新 id，例如 `commons-ten-v2`。冻结脚本拒绝覆盖已有的 testset.json。
2. **生成新清单**：
   - 合成集：`python -I scripts/benchmark.py freeze-synthetic --testset-id synthetic-v2`。
   - MegaDepth-X 真实 GPS 集：`python -I scripts/benchmark.py freeze-mdx --testset-id mdx-two-v2 --building <建筑>=<组件目录> ... --scene <建筑>=<场景名> ... --license-parquet <license.parquet> --description "..."`。
   - Commons 集：先 `python -I scripts/materialize_testset.py collect --pool 225 --max-screen 600 -- <fetch_commons_landmark.py 的参数>` 按选择规则收集每栋建筑，再 `python -I scripts/materialize_testset.py freeze --testset-id <id> --buildings <建筑> ... --fetch-root <收集输出> --mdx-root <参考模型> --licence-parquet <license.parquet> --description "..."`。新建筑先加进 `eval/landmarks.json`（Wikidata、Commons 分类、中心、半径）和 `eval/osm/`。
3. **把选择规则写进清单**：`selection`（候选来源、筛选顺序、保留规则和上限）、`created_with`（生成它的 git 提交，以及当时有没有未提交的文件）、`product_rule`、`access_notes`（当时哪些来源取不到）、`schema_notes`（每个字段的意思）。`commons-ten-v1` 是样板。
4. **证明可以复现**：在一个空目录里只凭 testset.json 重新物化一遍，所有照片的 sha256 都要对上（`commons-ten-v1` 的收集者做过：1,045/1,045 一致，10 个参考模型都核对通过）。
5. **只提交元数据**：testset.json 和 `outlines/*.geojson`（带 ODbL 署名和 sha256）。照片、点云、截图不进仓库；测试会检查测试集目录里没有图片和点云。
6. **重跑基线和噪声基线**：新测试集上先跑默认配置（`--repeat` ≥ 2，最好 5），再跑 `report`。旧测试集的结果不删，它们仍然可以和同一测试集上的新运行比较。
7. 跑一遍快速测试：`tests/test_benchmark.py::test_committed_test_sets_are_complete` 和 `tests/test_materialize_testset.py` 会检查已提交的测试集字段齐全、两个脚本读出来的一致。

## 10. 现状和已知局限

- **留出五栋**（圣家堂、匈牙利国会大厦、林肯纪念堂、印度门、悉尼歌剧院）只用冻结的提交跑（第 2 节）：基线 B0（`b0-test`）和冻结候选 F（`f-test`）各跑了两次重复，结果在 `eval/results/commons-ten-v1/`，结论见 [REVIEW_ADOPTION.md](REVIEW_ADOPTION.md)。林肯纪念堂、印度门、悉尼歌剧院两次都一个模型都没发布（模型被质量门以注册比例不够拦下，或者 SfM 找不到初始图像对），能和参考核对的模型只有 4 个（两次重复的中位数），只有匈牙利国会大厦一栋有 3 个以上；悉尼歌剧院的参考过不了轮廓核验（墙面中位距离 4.0 m），没有参考答案。所以留出集上的通过标准和标定都是 INSUFFICIENT。D31 的候选后来也在这五栋上跑了两次（`d31-test`，6.11 节）：林肯纪念堂、印度门第一次有了发布的模型，悉尼歌剧院仍然没有；能核对的模型到了 22 个，标定判得出了（水平位置不过，全是匈牙利国会大厦）。但 D31 的阈值是看了 F 在这五栋上的运行定的，这五栋对 D31 已经"烧掉"，那两次只是复现测量（第 2 节）。
- **照片多的建筑会撞到 1 小时的上限**：勃兰登堡门、圣母教堂的结果是"上传到一半"的状态。
- **"对 / 错"的判定本身有误差**：参考模型自己的偏航差 1–2°、尺度差几个百分点，相机离墙 80 m 时相当于 1.5–3 m，所以 1–2 m 量级的判定两个方向都可能错；近圆轮廓（比萨斜塔）上参考自己的旋转也定不下来。合成集用的是生成器真值，没有这个问题。
- **默认路径写死在脚本里**（第 3 节的注意）；**新重建器要改代码才能注册**（第 5 节）；**风格化参数不能通过配置调**（4.2）。
- **噪声估计偏小**（第 8 节）。
- **第 7 版的地理误差和覆盖都以参考为准**：参考自己偏航差 1–2°、尺度差几个百分点，所以 1–2 m 量级的地理判定两个方向都可能错；建筑局部的覆盖把 OSM 轮廓当作墙面（OSM 的误差、柱廊都算进 1.5 m 里）；只有带参考位姿的相机足够的模型能核对，参考覆盖不到的建筑（悉尼歌剧院的参考过不了轮廓核验；圣家堂的原始参考建好缓存以后删掉了，只从缓存读）只有 `cov_claimed_m` 和 `cov_unverified_m`。
- **分层还没做**：划分文件里写的"晚到的会话"和"没在 dev 出现过的摄影师"两层，`posediff` 和逐照片的指标还不分开报（持久地图那一步的工具）。
- **视觉定位的证据很少**（6.12 节）：B 里有参考位姿的定位位姿只有 11 个张次、6 张照片，全对；留一法里接受的 110 张有 24% 是错的。`run` 还不给运行里的定位位姿打分。
- **照片位姿的数字有几处要小心**：时间上限截掉、没上传的照片只在 `*_testset` 指标里算失败；参考位姿本身有 0.2–1.1 m 的墙面残差和 1–2° 的偏航误差，只影响 5 m / 10° 附近的判定；印度门的参考有 12 / 41 台相机看来被放到了拱门对面（第 7.7 节）；误差范围的标定是样本内的。
