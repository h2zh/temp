# 基准测试：怎么测、怎么比、怎么加新东西

本文是基准测试（`scripts/benchmark.py`）的使用说明，写给要跑测试、看结果、或者想换一种算法来比一比的人。

- 测出来的结果和结论在 [REAL_WORLD_EVAL.md](REAL_WORLD_EVAL.md)。
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
        │  benchmark.py compare / report
        ▼
对比表；eval/results/LEADERBOARD.md
```

| 文件 | 作用 |
|---|---|
| `eval/testsets/<id>/testset.json` | 冻结的测试集。`schema_notes` 字段逐项解释了其他字段 |
| `eval/configs/<id>.json` | 一组参数：用哪个重建器、改哪些产品阈值和算法参数 |
| `crowd3d/params.py` | 算法参数的唯一入口（`Params`），默认值就是产品出厂值 |
| `scripts/materialize_testset.py` | 按清单把 Commons 照片取到本地并逐张核对；维护者用它挑选和冻结新的 Commons 测试集 |
| `scripts/fetch_commons_landmark.py` | 礼貌地从 Wikimedia Commons 取照片和 EXIF（被上面的脚本调用） |
| `scripts/benchmark.py` | `prepare`、`run`、`compare`、`report`、`rescore`（不重跑 SfM，从已有运行补算照片位姿指标，`--restylize` 时重新风格化），以及维护者用的 `freeze-synthetic`、`freeze-mdx` |
| `crowd3d/photoposes.py` | 每张照片的拍摄位置、朝向和误差范围（第 7.7 节），`photo_pose_*` 指标用它 |
| `scripts/eval_real_landmark.py` | 单栋建筑的测试流程和**全部指标的定义**（`METRICS`、`METRIC_SCHEMA_VERSION`） |
| `scripts/stylize_model.py`、`crowd3d/stylize.py` | 风格化（低多边形"玩具"模型）和它的配准门；基准对每个已发布的模型各跑一次 |
| `eval/results/<id>/<运行名>.json` | 一次运行的结果（只有数字和路径） |
| `eval/results/LEADERBOARD.md` | `report` 生成的排行榜，不要手改 |

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
- 摄影师：按 `materialize_testset.normalize_author` 认人，14 位摄影师两边都有，拍了留出集 513 张里的 53 张；2 张没有作者信息，不算见过也不算没见过。逐张照片的指标在留出集上另报没见过的摄影师的 458 张。
- 已知的污染（每次报留出结果都要附上）：结构规则是在目录里 13 个地标的 OSM 快照上定的，包括这五栋（G3 来自林肯纪念堂的台地、G10 来自印度门）；照片位姿 EXIF 位置 σ 的标定照片包括这五栋（第 7.7 节）；印度门的参考有 12/41 台相机看来放到了拱门对面，它的对错单独报，不进合计。
- 划分文件记着它划的是哪个版本的测试集（testset.json 的 sha256），每栋建筑必须正好在 dev、test 之一里，否则 `run` 和 `rescore` 拒绝。结果的 `testset.split` 记下划分的 id、路径、sha256 和两个名单；没有划分文件的测试集（synthetic-v1、mdx-two-v1）记 `null`，整个算开发集。一个测试集有多个划分文件时要用 `--split` 指定。

**OSM 分件快照**（`eval/osm/<slug>.parts.osm`，目录里 13 个地标）：从 2026-10-22 下载的 OSM API 0.6 整块快照（地标周围的一个框）裁出来，只留 `building:part` 的线和关系、轮廓文件指定的主体、它们的成员线和节点，去掉 user、uid、changeset 和节点标签；整块快照最新元素的时间、下载日期和裁剪规则写在根元素上，`structure.parse_osm` 读回来当快照元数据。裁剪工具 `scripts/trim_osm_parts.py` 只有在裁后的文件建出和整块完全相同的结构（登记表、报告、丢掉的分件、主体标签、每块积木的三角形）时才写，并在 `eval/landmarks.json` 里记 `osm_parts`：文件、sha256、来源、下载日期、最新元素、许可（ODbL 1.0）和署名。OSM 本身不改，裁剪只是摘出。有快照的建筑（mdx-two-v1 和 commons-ten-v1 的全部建筑），快照的 sha256 进 `inputs_sha256`（第 6.1 节）；现在的测试流程还不读分件，结构接进基准之后才用。

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
- 现有两个配置：`default.json` = 产品默认值（基线）；`fast.json` 只把处理图长边从 1600 px 降到 1024 px。

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
- 风格化有自己的时间预算（`STYLE_TIME_CAP_S` = 1800 s，从风格化开始算），没来得及做的模型记在 `style_incomplete`。`--no-stylize` 关掉风格化，`style_*` 指标为空。
- `--resume`：中断之后（例如容器重启）复用 `--work-root` 里已经跑完的建筑。只有运行键完全相同才复用：代码指纹（`crowd3d/`、`scripts/`、`eval/configs/` 的 sha256）、配置、测试集、实际输入和选项。改了任何代码，所有建筑都会重跑。
- 实际输入 `inputs_sha256`：用到的照片和它们的 sha256、轮廓、参考模型各文件的状态、OSM 分件快照的 sha256（有快照时；没有快照的合成集和以前记的一样）。参考答案已经缓存时，参考模型的状态取建缓存时记下的那份，所以缓存建好后删掉参考模型，`inputs_sha256` 不变。
- 磁盘守卫：`--work-root` 所在的盘剩余不到 3 GiB 时，不开始新的建筑运行，整个命令停下（`--resume` 能复用的已完成运行不受影响）。已经跑完的建筑留着，腾出空间后加 `--resume` 接着跑。
- 结果的 `testset.split` 记下测试集的开发 / 留出划分（第 2 节）。
- `--note`：一段说明，存进结果，也显示在排行榜上。写清楚这次运行和别的运行有什么不同、有没有跳过的建筑。
- 每栋建筑的数据目录、`summary.json`、`summary.md`、俯视图 `topdown.png`、风格化的 GLB 都在 `--work-root/<测试集>/<运行名>/<建筑>/` 下，不进仓库。
- 结果里的 `provenance` 记下 git 提交、有没有未提交的改动、Python/pycolmap/OpenCV/numpy/Pillow 的版本、CPU 数、开始时间、总用时、命令行，以及重建输出版本 `recon_output_version`（见 6.4）。
- 参考答案（第 7.3 节的参考位姿和"轮廓校正参考"）每栋建筑只算一次，缓存在 `--reference-cache`（默认 `<work-root>/<测试集>/references/<建筑>.json`），`run` 和 `rescore` 共用。缓存键是测试集 sha256（照片、参考模型的图像对应和稀疏文件哈希、轮廓）加参考版本 `REFERENCE_VERSION`，不含测试流程版本：只改指标不会让缓存失效。有缓存的建筑不再读参考模型（MegaDepth-X 的 sparse），所以缓存建好后参考模型可以删掉。缓存里是参考相机（ENU 位置和朝向，全精度）、各分量的报告、校正后的参考点（只取通过验证的分量，每栋最多 100,000 个，厘米精度）、参考自己的误差 `sigma_ref`（偏航 1°、尺度 2%；合成集的生成真值为 0），以及 `gt_model_check` 用的参考位姿。

### 6.2 比（`compare`）

```bash
python -I scripts/benchmark.py compare eval/results/synthetic-v1/v1-merged.json \
    eval/results/synthetic-v1/my-change-1.json [--noise <带噪声块的结果>] [--json out.json] [--fail-on-worse]
```

- 两边测试集 sha256 或指标版本不同：拒绝（`--force` 才比）。
- 某栋建筑两边的 `inputs_sha256` 不同（有照片变了或缺了）：单独列出，不进汇总。分件快照进指纹之前记的结果，有快照的建筑和新运行比都会显示输入不同；用 `rescore` 补算一遍，结果就按现在的方式记指纹。
- 每个指标的差值超过 **max(该指标的最小变化量, 相对下限 × 旧值, `--noise-factor` × 记录的运行间极差)** 才算"显著"。极差取自两个结果本身、`--noise` 指定的文件、同一测试集下所有带噪声块的结果中最大的那个，**但只用指标版本相同的结果**。
- 方向来自指标定义：↑ 越大越好、↓ 越小越好、"-" 只是背景信息（只报"变了"）。显著变差的个数和显著变好的个数单独统计，`--fail-on-worse` 在有显著变差时返回 1，可以接进 CI。
- 一个指标一边有、一边没有（例如某次运行多发布了一个模型）也算显著，除非重复运行记录过它本来就时有时无。
- 风格化被时间预算截断的建筑，`style_*` 指标记为 `incomplete`，不算变化。

### 6.3 出排行榜（`report`）

```bash
python -I scripts/benchmark.py report       # 重新生成 eval/results/LEADERBOARD.md
```

排行榜列出全部运行的汇总值和说明、每个配置最新一次运行的分建筑结果、照片位姿一节（每次带 `photo_pose_*` 的运行，分建筑列出）、各噪声运行的极差，以及完整的指标定义。它是生成的文件，不要手改。

### 6.4 补算（`rescore`）

照片位姿指标（第 7.7 节）只看一次运行留下的东西：数据目录里的 SQLite（照片、已发布模型和每台相机的位姿）、`summary.json` 里的照片行和风格化接受的轮廓吸附。所以已经跑过的运行不用重跑 SfM，就能补算：

```bash
python -I scripts/benchmark.py rescore --testset commons-ten-v1 \
    --from eval/results/commons-ten-v1/v1-after-fixes.json --name v1-poses [--restylize]
```

- `--restylize`：先用现在的代码把运行保留下来的每个已发布模型重新风格化一遍（和 `run` 里一样，同一份参考答案），`style_*` 指标和照片位姿都用新的风格化结果；风格化的 GLB 和每次重复的 `style.json` 写在 `<work-root>/<测试集>/<运行名>/<建筑>/<重复>/`，源运行目录不动。同样的代码、同样的参考，重新风格化得到的记录和运行时的一样（时间和路径除外；`tests/test_eval_real.py` 的慢测试核对）。
- 重建输出版本：`crowd3d/reconstruct.py` 的 `RECON_OUTPUT_VERSION` 标记一次重建作业对同样的上传留下什么（稀疏模型、PLY、模型 JSON、质量门结论）。能改变这些输出的改动（SfM 选项、选模型、地理配准、相机检查、导出、质量门、作业规划、照片接收、`Params`/`Settings` 默认值）都要把它加 1；`tests/test_recon_golden.py` 用一个固定的合成稀疏模型核对重建之后的每一步和传给 COLMAP 的选项，输出变了而版本没变就失败。`rescore` 拒绝补算版本不同的运行（`--allow-recon-diff` 才算，结果记下它算的是哪个版本）。所以只改风格化或指标的提交之后，旧运行的产物还能接着补算。这个版本出现之前的结果按提交对应（`LEGACY_RECON_OUTPUT_VERSION`）：v1-merged（0a49b22）和 v1-after-fixes（1d6055b、3bbb0dd）是版本 1，v0 的运行不是。

- 读 `--from` 结果的 `artefacts_dir`，对每栋建筑的每次重复（r1、r2……）各算一次，记录中位数，和 `run` 一样。其余指标从源结果原样拷过来；结果里 `provenance.rescored_from` 记下源结果的名字、路径和 sha256。
- `inputs_sha256` 按现在 `run` 的方式重新算（含分件快照），所以补算过的旧运行能和新运行在同样的输入上比；本地的照片、轮廓或参考和源运行用的不一样时（新旧两种指纹都对不上源结果）会警告。`testset.split` 记现在的划分。
- 数据目录只读：SQLite 先复制到临时目录再打开（`Store` 打开时会给旧库加列）。
- 源结果里没有的建筑（commons-ten-v1 只跑过核心五栋）写 `skipped (no run in the source result)`，不打分。
- 参考答案用 6.1 的缓存（`--reference-cache`，和 `run` 共用）。缓存不存在、参考模型又不在本地的建筑没有参考答案，对错类指标为空。
- 测试集分母：每栋有 SfM 运行的建筑，`rescore` 按测试集重新列出流程本该上传的照片（文件在、有 EXIF GPS；`upload_all` 的合成集是全部），和源运行 `summary.json` 的漏斗核对（对不上会警告）。时间上限截掉、没上传的照片在 `*_testset` 指标里算失败，每次重复的 `poses.runs[]` 记下 `testset_photos`、`not_uploaded`、`not_uploaded_gt`。
- 结果文件同名已存在时要加 `--overwrite`。

## 7. 每个指标是什么意思

全部指标定义在 `scripts/eval_real_landmark.py` 的 `METRICS` 里，排行榜末尾有英文原文。现行版本 `metric_schema_version` = 6。

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

### 7.4 运行时间

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `wall_time_s` | ↓ | sum | 这栋建筑整个运行的墙钟时间（不含建参考答案的时间：它是评测的输入，有缓存） |
| `jobs_time_s` | ↓ | sum | 重建作业时间之和 |

机器同时在跑别的东西时这两个数会被放大，比较时要看运行说明。

### 7.5 风格化（`style_*`）

先说清楚三个词：

- **可参评**（eligible）：已发布、并且完成了完整地理配准（`georeg_level = full`）的模型。只靠罗盘和粗估尺度的 `prior_only` 模型照样风格化，但不计入通过率，而且必须整栋素模。
- **配准门**：吸附到 OSM 轮廓以后，墙点离轮廓足够近、逐边核验通过、并且没有任何歧义或倾斜等拒绝理由（docs/STYLIZED_MODEL.md §3.2）。没过门的模型整栋只出素模（"待解锁"），一面墙都不上色。
- **对 / 错**：过了门的模型，再拿轮廓校正后的参考位姿核对（至少 3 台相机有参考位姿才下结论）。四条都满足才算对：相机水平中心离参考的中位数 ≤ 1.5 m；还需要的转角 ≤ 5°；尺度差 ≤ 10%；墙点还要挪动的 90 分位 ≤ 1.5 m（取二维和三维里大的那个，倾斜也算）。

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `style_models_stylized` | - | sum | 跑了风格化的已发布模型数 |
| `style_models_eligible` | - | sum | 其中可参评的 |
| `style_registration_passed` | ↑ | sum | 可参评且过了配准门的 |
| `style_registration_pass_rate` | ↑ | pooled | 过门数 / 可参评数，跨建筑按模型合并 |
| `style_registration_checked` | - | sum | 过了门、又有 ≥ 3 台参考相机可以核对的 |
| `style_registration_right` | ↑ | sum | 其中对的 |
| `style_registration_wrong` | ↓ | sum | 其中错的：照片会贴到错的墙上或墙的错的位置。**这是最要紧的指标，目标是 0** |
| `style_rejected_placed_right` | ↓ | sum | 被门拒了、但单靠地理配准其实已经放对了的：本来可以上色却成了素模，是"宁可不上色"的代价 |
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
| `style_walls_total` | - | sum | 每栋建筑"最佳模型"（上色墙最多的那个，其次看观测面积）的墙数 |
| `style_walls_coloured` | ↑ | sum | 最佳模型里上了照片颜色的墙数（已解锁 + 部分解锁）；其余是素模 |
| `style_walls_unlocked` | ↑ | sum | 其中"已解锁"的（掩码 ≥ 60%，且 ≥ 3 台相机各看到 ≥ 15%） |
| `style_observed_area` | ↑ | mean | 最佳模型里拍到的墙面积占比（不含补洞） |
| `style_roof_observed_area` | - | mean | 拍到的屋顶面积占比 |
| `style_fill_m2` | - | sum | 用墙自己的实测底色补上的小洞面积（D12） |
| `style_qa_failed` | ↓ | sum | 没通过诚实检查（诚实 QA、`audit_model`、`audit_glb`）的模型数；**必须为 0** |
| `style_provenance_violations` | ↓ | sum | 诚实检查的违规总数（没拍到的地方出现了照片色、没有来源的面等）；**必须为 0** |
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

所以 v0 结果（版本 2）里的"错误通过"是按最松的规则判的；按版本 5 判，错的只会更多。

### 7.7 照片位姿（`photo_pose_*`）

**要回答的问题**：在三维模型上画出每张照片是在哪儿拍的、朝哪个方向拍的（"大致方位"），能画出来的有多少张、画得对不对。**没法确定方位的照片算失败。**

**每张照片用哪个来源**（`crowd3d/photoposes.py`；先有先用）：

1. `sfm`：照片被 SfM 注册进一个**已发布**的模型、相机没被标记——就是地标页现在显示的估计位姿（同一张照片在多个模型里时取最新的，完整模型优先）。模型的轮廓吸附被接受时，相机跟着吸附一起移动。
2. `exif`：照片自己 EXIF 里的 GPS 位置加罗盘方向（`GPSImgDirection`，按标注原样用）。
3. `gps`：有 GPS、没有罗盘、SfM 也没用上：只知道位置、不知道朝向，**算失败**。不会假设"照片一定朝着建筑"。
4. 被拒的上传：没有位姿，**算失败**。

产品没有视觉定位器（[VISUAL_LOCALIZATION.md](VISUAL_LOCALIZATION.md)），所以没有 `loc` 来源，`photo_pose_loc` 恒为 0（指标留着，和 #87 的版本 6 定义一致）。

**误差范围**（`sp` 米、`sy` 度，2σ；规则在 commons-ten-v1 和 mdx-two-v1 的运行上标定，位置和朝向各自约覆盖本来源 85–90% 的误差，样本内）。位置和朝向**同时**落在范围里的比例更低，而且误差有重尾（地理标签标错、罗盘差约 90° 或 180°），多大的圈都盖不全：`photo_pose_*uncertainty_coverage` 按来源报实测的覆盖率，结果的 `provenance.photo_pose_uncertainty_rule` 记着算它时用的 σ 规则。标定用的照片包括 commons-ten-v1 没跑过 SfM 的五栋（它们当时按"仅 EXIF"打过分），所以 EXIF 的 σ 对这五栋不算样本外。没用 GPS 配准的模型那一行是第二轮修正加的：mdx-two-v1 亚琛唯一发布的模型就是这种（4 台相机，3 个一致的 GPS 只跨 6.9 m，偏航来自地标方向），它的 4 张参考照片偏了 23.6–24.8 m，原来却画着 15 m 的圈。30 m 的依据：这种模型的位置只靠几张照片 GPS 的中位数，和手机自己的 GPS 一样粗（commons-ten-v1 十栋 527 张参考照片的 EXIF GPS 只有 72% 落在参考位置 30 m 内，几张取中位数会好一些）；0.6 对应 `_fallback_scale` 在真实建筑上偏大 1.5–2.4 倍（1 − 1/2.4 ≈ 0.58）。有参考位姿的这类相机三个测试集一共只有 10 台，属于样本内的小样本：

| 来源 | σ 位置 | σ 朝向 |
|---|---|---|
| `sfm` | max(15 m, 配准 RMSE, 相机离中心距离 × sin\|GPS 偏航 − 罗盘偏航\|) / 2 | max(√(偏航 σ² + 1), \|偏航冲突\| / 2, 10°)；配准自己报的偏航 σ（3–5°）不够，真实模型偏过 15–21° |
| `sfm`，吸附已接受 | ≤ 3 m | ≤ 3° |
| `sfm`，模型**没能用 GPS 配准**（`georegistered` 为假：偏航只来自罗盘或地标方向，尺度是估的，位置取 GPS 偏移的中位数），没吸附 | 圆至少 hypot(30 m, 0.6 × 相机离该模型相机中位点的距离)，再 / 2；记录里标 `approx` | 同 `sfm` |
| `exif` | 真北标注（`T`）max(20 m, 手机报的 GPS 误差)，其他 max(30 m, 手机报的 GPS 误差)；只有位置（`gps`）同"其他" | 真北标注（`T`）15°，其他（磁北 `M` 或没标）30°：磁北标注常差约 90°，15° 内只有 47%（真北 85%） |

EXIF 位置 σ 是第三轮修正改的（原来一律 12 m）：按 EXIF 画出来、有参考位姿的 51 张照片（commons，核心五栋的 r1 加仅 EXIF 五栋，每张只算一次）只有 37 张位置落在 24 m 的圈里、34 张位置和朝向都在。180 张带罗盘标注、有参考位姿的照片的原始 EXIF GPS（不管 SfM 有没有定位上）：`T` 标注的 90 分位 23 m，`M` 标注 60 m；真正按 EXIF 画出来的是 SfM 没定位上的那些，更差（`T` 的 85 分位 38–40 m）。σ 取 20 / 30 m 后，这 51 张里位置落在圈里的 43 张（84%）、位置和朝向都在的 40 张（78%）。剩下的是重尾：林肯纪念堂两张的 EXIF 位置离参考 109 m、353 m（参考相机在纪念堂里），圣家堂一张 182 m；印度门 3 张看起来是参考自己错了：41 台参考相机里有 12 台离自己 EXIF GPS 关于地标中心的对称点很近、离 EXIF GPS 本身很远（中位 8 m 对 135 m），MegaDepth-X 看来把它们放到了这座前后几乎一样的拱门的对面。带罗盘标注又报了 `GPSHPositioningError` 的只有 mdx-two-v1 洗礼堂的 15 张（commons 的元数据里没有这个字段），14 张误差 ≤ 17 m，1 张 51 m（它报的是 5 m）：报的精度只用作下限，不用来缩小圈。

**怎么算对**：有参考位姿（合成集用生成器真值；真实照片用第 7.3 节的轮廓校正 MegaDepth-X 参考，只用验证通过的分量）的照片，水平位置差 ≤ 10 m **且**朝向差 ≤ 15° 算对。10 m 能保证相机站在建筑的正确一侧、大致正确的位置（这些建筑宽 30–150 m）；15° 是产品自己的罗盘 σ，也不到手机视场角（约 60°）的一半，画出的视锥仍然对着同一片立面。另外报更严的 5 m / 10°。

| 指标 | 方向 | 汇总 | 意思 |
|---|---|---|---|
| `photo_pose_determined` / `_rate` | ↑ | sum / pooled | 位置和朝向都确定了的上传照片张数，和它占全部上传的比例（分母含被拒的） |
| `photo_pose_sfm`、`_loc`、`_exif` | - | sum | 各来源的张数 |
| `photo_pose_gps_only` | ↓ | sum | 有 GPS、没朝向：失败 |
| `photo_pose_gt`、`_gt_determined` | - / ↑ | sum | 有参考位姿的上传照片（含被拒的），其中确定了的 |
| `photo_pose_correct` / `_rate` | ↑ | sum / pooled | 有参考位姿的照片里，确定了**而且**对的；比率的分母是全部有参考的照片，没确定的和错的都算失败。**这是最诚实的通过率** |
| `photo_pose_wrong` | ↓ | sum | 确定了但错了 |
| `photo_pose_correct_strict` / `_rate` | ↑ | sum / pooled | 按 5 m / 10° 算 |
| `photo_pose_pass` / `_rate` | ↑ | sum / pooled | 确定了、而且没被证明是错的 / 全部上传。没有参考位姿的照片无法核对，按通过算，所以偏乐观 |
| `photo_pose_testset_photos`、`_not_uploaded`、`_gt_testset` | - / ↓ / - | sum | 测试集里流程本该上传的照片（文件在、有 EXIF GPS；合成集全部）、其中被时间上限截掉没上传的、测试集里有参考位姿的（上传没上传都算） |
| `photo_pose_determined_rate_testset`、`photo_pose_correct_rate_testset` | ↑ | pooled | 同上两个比率，但分母换成整个测试集：没上传的照片也算失败。**每个测试集、每栋建筑的通过率按这两个报** |
| `photo_pose_pass_rate_low_testset` / `_high_testset` | ↑ | pooled | 全部测试集照片上的通过率下界（确定且对 / 全部：没参考的都算失败）和上界（确定且没被证明是错的 / 全部：没参考但确定了的算通过）。有参考位姿的照片偏向 SfM 定位到的那些，所以 `correct_rate` 要和这对上下界一起看 |
| `photo_pose_sfm_gt`、`_sfm_correct`、`_sfm_correct_rate`；`photo_pose_exif_…` | - / ↑ | sum / pooled | 按来源分开的对错 |
| `photo_pose_position_error_median_m`、`photo_pose_heading_error_median_deg` | ↓ | median | 确定了的参考照片的误差中位数 |
| `photo_pose_within_uncertainty` / `photo_pose_uncertainty_coverage` | - | sum / pooled | 位置误差和朝向误差**同时**落在画出的误差范围（`sp`、`sy`）里的比例（每样各自约 85–90%，同时在就更低） |
| `photo_pose_sfm_within_uncertainty` / `photo_pose_sfm_uncertainty_coverage`；`photo_pose_exif_…` | - | sum / pooled | 同上，按来源分开（结果的 `provenance.photo_pose_uncertainty_rule` 记着算它时用的 σ 规则） |

**在产品里**：还没有。基准只量这些位姿；把它们画给用户看的接口和查看器要先定隐私规则（README"下一步"第 1 条）。

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
- **只用同一指标版本的噪声**。现行版本 5 下，只有 `v1-after-fixes` 和 `v1-merged` 各自的 2 次重复可用。指标版本升级后，要重新记录噪声基线。
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

- **commons-ten-v1 只记录了核心五栋**（洗礼堂、亚琛、勃兰登堡门、比萨斜塔、圣母教堂）。另外五栋（圣家堂、匈牙利国会大厦、林肯纪念堂、印度门、悉尼歌剧院）还没跑：两次重复的核心五栋就要约 5.5 小时的 COLMAP 时间，容器大约每 3 小时重启一次。命令见 [REAL_WORLD_EVAL.md](REAL_WORLD_EVAL.md) 第 7 节。这五栋现在是留出集，只用冻结的候选跑（第 2 节）。
- **照片多的建筑会撞到 1 小时的上限**：勃兰登堡门、圣母教堂的结果是"上传到一半"的状态。
- **"对 / 错"的判定本身有误差**：参考模型自己的偏航差 1–2°、尺度差几个百分点，相机离墙 80 m 时相当于 1.5–3 m，所以 1–2 m 量级的判定两个方向都可能错；近圆轮廓（比萨斜塔）上参考自己的旋转也定不下来。合成集用的是生成器真值，没有这个问题。
- **默认路径写死在脚本里**（第 3 节的注意）；**新重建器要改代码才能注册**（第 5 节）；**风格化参数不能通过配置调**（4.2）。
- **噪声估计偏小**（第 8 节）。
- **照片位姿的数字有几处要小心**：时间上限截掉、没上传的照片只在 `*_testset` 指标里算失败；参考位姿本身有 0.2–1.1 m 的墙面残差和 1–2° 的偏航误差，只影响 5 m / 10° 附近的判定；印度门的参考有 12 / 41 台相机看来被放到了拱门对面（第 7.7 节）；误差范围的标定是样本内的。
