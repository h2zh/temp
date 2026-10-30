# Crowd3D：众包打卡照片 → 地标三维模型（原型）

不同的用户在著名建筑前打卡，从不同角度上传照片。系统自动判断每张照片是从建筑的哪一侧、朝哪个方向拍的。照片还不够时，先重建能拍到的那一面（**立面模型**）。照片覆盖了大部分方向后，再重建整栋建筑（**全景模型**）。模型以地标中心为原点，单位是米，北向对齐。

在点云之上，原型还能生成一个**风格化的"玩具"模型**（GLB）。它走"结构优先"的路线，参照 Hover 的做法：

- **框架先取自地图。** OpenStreetMap 的 Simple 3D Buildings 分件（`building:part`）变成参数化积木。每块积木带一个屋顶原语（平顶、单坡、双坡、四坡、攒尖、穹顶、洋葱顶……），每个参数都记下来源。
- **照片按规则修改地图上的值。** 证据有三路：天空轮廓线、SfM 点、三维直线。
  - 至少两路一致时自动采用（D15）；改高度时还要读到这块积木自己的屋顶（D27）。
  - 照片量出立面的凹凸（D24）。
  - 证据足够强时把 OSM 的高度往上纠正、加上倾斜、给缺高度的部分补出体块（D25）；每处纠正都可以撤销。
- **颜色只来自照片真正拍到的部分。** 没拍到的地方显示为素模（灰色斜纹）。画风可以整套替换。

这是一个只用 CPU 的最小原型。后端是 Python（FastAPI + SQLite + pycolmap），前端是不需要构建的原生 JS 页面。另带一个能生成"真值"的合成数据生成器，用于测试和演示。

相关文档：
- 完整的技术架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，第 11 节逐个模块对照原型和生产架构。
- 风格化模型见 [docs/STYLIZED_MODEL.md](docs/STYLIZED_MODEL.md)。
- 结构优先的设计、判定规则和实测结果见 [docs/STRUCTURE_FIRST.md](docs/STRUCTURE_FIRST.md)，产品里的结构见其中 §2.9。
- 画风见 [docs/ART_DIRECTION.md](docs/ART_DIRECTION.md)。
- 外部评审采纳了什么、没采纳什么、测出了什么，见 [docs/REVIEW_ADOPTION.md](docs/REVIEW_ADOPTION.md)。

> **现状**：上传、校验、重建、发布这条链路已经接进服务和网页。结构优先也接进了作业系统：有 OSM 分件快照的地标，每个发布的模型会自动生成一个 GLB，地标页链到它。但今天每个真实众包模型都过不了配准门，产品又只用一种可商用的天空掩码，所以产品给出的都是 OSM 分件的素模：没有照片颜色，也没有照片量出的数字。详见"原型的局限"。

## 快速开始

需要 Python ≥ 3.11（开发时用 3.13）。运行 Web 测试还需要 Node ≥ 22（开发时用 v22），可选：测试用到了 `mock.timers` 的 Date 模拟和 `node --test` 的通配符，旧版本跑不了。

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q -m "not slow"   # 快速测试，不需要真实照片
python -m pytest -q                 # 加上慢测试（真实 COLMAP、端到端验收、风格化作业端到端）
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

终端里会打印文字版罗盘、作业状态和模型质量。最后给出查看地址，例如 `http://127.0.0.1:8000/landmark.html?id=demo-building`。在浏览器里可以看到罗盘、共同完成者、作业列表、模型列表和 three.js 点云查看器。首页 `http://127.0.0.1:8000/` 是地图和地标列表。

常用参数：

- `python -m crowd3d.api`：`--data-dir DIR`（默认 `./data` 或 `$CROWD3D_DATA_DIR`）、`--reconstructor fake`（不跑 COLMAP，秒出假模型，适合调界面）、`--host`、`--port`。
- `python scripts/demo.py`：`--users N`（参与的游客数）、`--size 800x600`（照片越大越慢，默认 640x480）、`--seed 1`（换一组照片；同一组照片重复上传会被判为重复）、`--landmark ID`（新建或使用另一个地标）、`--server URL`、`--photos DIR`。
- 自己生成照片，再从网页手动上传：`python -m crowd3d.synthetic --out /tmp/photos --size 640x480`。照片的 EXIF 带 GPS 和指南针朝向，目录里有 `manifest.json` 记录真值。

> 网页从 CDN 加载 Leaflet、three.js 和 OSM 地图瓦片，所以浏览器需要能访问外网。

### 风格化模型

**服务会自动生成。** 地标满足下面三条时，每发布一个 COLMAP 模型，作业系统就排一个风格化作业：
- 地标目录（`Settings.structure_catalog`，默认 `eval/landmarks.json`）里有它的 OSM 分件快照和轮廓。目录里的 13 个地标都有。
- 地标 id 等于目录里的 slug（例如 `florence-baptistery`），或者在 `Settings.structure_bindings` 里绑定了 slug。
- 目录里的中心在地标的打卡半径内。

GLB 放在 PLY 旁边。地标页的模型列表里会出现"风格化模型 · 素模（照片未对齐）"或"风格化模型 · 风格化（照片已对齐）"的链接，生成中显示"风格化生成中"。演示地标 `demo-building` 不在目录里，`--reconstructor fake` 也不写稀疏地图，所以演示不会生成 GLB。

**也可以用脚本处理单个模型。** 需要两样输入：
- 一个已重建的作业，或者 MegaDepth-X 格式的参考场景（`--scene DIR`）。作业目录默认会被清理。`keep_work_dirs` 没有命令行参数，也没有环境变量，要在代码里用 `Settings(keep_work_dirs=True)` 启动服务。基准（`scripts/benchmark.py`）默认会保留作业目录。
- 地标在 `eval/landmarks.json` 里的条目：坐标、OSM 轮廓文件和分件快照（`osm_parts`）。外轮廓拉伸时，高度标签来自轮廓 GeoJSON；结构优先时，来自分件快照里每个分件的标签。条目里只有 Wikidata 编号，脚本用不到 Wikidata 的高度。

输出必须写到仓库外（或被 git 忽略的 `data/`），因为贴图来自有许可要求的照片：

```bash
python scripts/stylize_model.py --job-dir data/work/<作业> --meta data/models/<地标>/<模型>.json \
    --landmark florence-baptistery --out data/stylized
```

地标有分件快照时（目录的 `osm_parts`，或用 `--osm-parts FILE` 指定别的快照），脚本默认走结构流水线 `crowd3d.structure_pipeline`。它和服务的风格化作业是同一个入口：
- OSM 分件做体块；
- 照片的配准被接受时，用 A+R 天空掩码、SfM 点和三维直线做 D15 判定；
- D25 按 `--d25` 执行：`auto`（默认）只在有两种可商用天空掩码时跑，所以今天不跑；`survey` 只量不改；`off` 不跑。

`--no-structure` 退回外轮廓拉伸。摘要 JSON 的 `structure_run` 写明跑了什么、没跑的原因。天空掩码和线段缓存在 `<out>/cache`。

crowd3d 只认 A+R 一种天空掩码（`structure_pipeline.PRODUCT_MASKS`），别的掩码名一律拒绝，没有绕过的开关。研究用的掩码（例如 Dm+R）只在仓库外的研究脚本里用，因为它们的许可没有确认可以商用（见 [docs/REVIEW_ADOPTION.md](docs/REVIEW_ADOPTION.md) 第 1 节）。

`--snap-from <旧摘要或 GLB>` 复用旧版本的吸附，重新发布时模型不漂移。流水线模型（`--job-dir`）要先通过现在的吸附门重新核对；参考场景（`--scene`）存下的吸附原样复用。

输出包括 GLB、摘要 JSON 和调试图；有 D25 纠正时另有一份 OSM 修正建议（`<slug>-osm_fix_suggestions.json`）。

查看方法：
- 查看器 `web/stylized.html?model=<GLB 的地址>&theme=<画风>` 只接受同源地址或相对路径。可选画风是 `default`、`storybook`、`dusk`、`blueprint`。
- 服务生成的 GLB 在 `/files/models/<地标>/<模型>.glb`，地标页直接链过去。
- 脚本写出的 GLB 服务不提供（`/files/models` 只读 `data/models`）。可以在仓库根目录起一个静态服务器（例如 `python -m http.server`），再打开 `/web/stylized.html?model=/data/stylized/florence-baptistery.glb`。

## 截图

以下截图来自真实服务（COLMAP 重建）：先运行 `scripts/demo.py`，再在网页上传 2 张合格照片和 3 张不合格照片。截图环境不能访问外网，所以地图底图是灰色占位。截图比风格化链接和"GPS 对齐残差"的改名都早。

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
上传 ─▶ 读 EXIF ─▶ 校验 ─▶ 入库（扇区）─▶ 覆盖规划 ─▶ 重建作业（COLMAP）─▶ 地理配准 ─▶ 发布质量门 ─▶ PLY + 元数据 ─▶ 网页
                                                                                          │
                               （风格化作业，在子进程里）OSM 分件框架 + 照片证据 + 风格化 ◀─┘ ─▶ GLB ─▶ 风格化查看器
```

### 点云链路（已接进服务）

1. **读 EXIF**（`exif.py`）：GPS、指南针朝向（`GPSImgDirection`）、焦距、机型、拍摄时间。
2. **校验**（`validation.py`）：所有不通过的原因一起返回，不会遇到第一个就停。拒绝码有：
   - `no_exif`、`no_gps`、`no_landmark`、`too_far`（超出地标半径）
   - `facing_away`（指南针朝向偏离"相机→地标"方向超过 60°）
   - `too_small`（短边 < 480 px）
   - `too_blurry`（拉普拉斯方差，按 800 px 归一化）
   - `duplicate`（dHash 汉明距离 ≤ 4）
   - `unreadable_image`
   - 另外，`ingest.py` 在两种情况下给出 `too_large`，并且只返回这一个码：文件超过 25 MB；或者不是 JPEG、超过 5000 万像素。
   - 警告码不会导致拒绝：`no_heading`（没有指南针，无法检查朝向）、`magnetic_heading`（朝向是磁北，没有做磁偏角修正）。
3. **扇区**（`coverage.py`）：地标四周分成 8 个罗盘扇区（N、NE……NW）。照片归入"地标中心→拍摄者"方位所在的扇区，也就是拍摄者看到的那一面。
4. **覆盖规划**（`coverage.plan_jobs`，纯函数）：
   - 某个扇区有 ≥ 4 张合格照片时，触发**立面作业**，输入是该扇区的照片，加上相邻扇区 10° 以内的照片。
   - 总数 ≥ 24 张、就绪扇区也够时，触发**全景作业**。
     - 默认要 6 个就绪扇区。
     - 照片满 48 张后，没有照片的方向算作拍不到，要求降为 max(⌈0.75A⌉, A−2, 2)，A 是有照片的扇区数。
   - 之后照片继续增长，立面每多 3 张、全景每多 8 张就重建一次。失败的作业也计入，要等再增长够了才会重试。
5. **作业**（`jobs.py`）：单个后台线程按顺序执行（SfM 很吃 CPU）。
   - 作业做完后会重新规划一次，所以作业运行期间新到的照片不会被漏掉。
   - 服务重启时，未完成的作业会重新排队。但进程已经在同一个作业中途退出过 2 次的，直接记为失败，不再重跑。
   - 只有一条替换规则（`jobs.superseded_by`）：旧作业晚完成时，不会盖掉同一目标更新的已发布模型。已发布的模型含有这次没有的照片、而且这些照片上传得更晚时，这次的模型存为未发布，作业以"已完成"结束并写明原因。
   - 风格化作业排在重建作业后面，见下一节。
6. **重建**（`reconstruct.py`）：
   - pycolmap 提取 SIFT 特征。不超过 60 张时做穷举匹配，更多时按 GPS 做空间匹配。
   - **特征缓存**（`landmark_map.py`，`Params.feature_cache`，默认开）：每个地标一个缓存。它按处理图的 sha256 原样存下 COLMAP 写过的特征、匹配和两视图几何。作业开始前把已有的行写进自己的数据库，COLMAP 只算新的。建图读到的数据库和不用缓存时逐字节相同（验证了 46 个作业），核心五栋的作业 CPU 少了 59.7%。
   - 然后做增量建图，而且不止建一次：不超过 30 张时用 3 个种子各建一次；更多时，只有结果可疑（超过 5% 的相机被标记，或平均罗盘误差超过 20°）才再建一次。最后留下未标记相机最多的模型。建图每次仍从零开始。从目标上一个模型续建（`Params.map_continue`）做了，但开发集验证没过，默认关。
   - 少于 12 张、并且每台相机都有 35 mm 等效焦距时，固定使用 EXIF 焦距。原因是一个近似平面的立面只有几张照片，定不准焦距。
7. **地理配准**（`reconstruct.georegister`，纯 numpy）：
   - 用相机的右向量和相机中心的分布估计重力方向。直接平均"上"向量会因为仰拍偏差约 10°。
   - 用鲁棒的 2D 相似变换把相机中心对齐到 GPS，偏航角再按反方差和指南针融合。没有指南针时，用"伪罗盘"：假设每张照片都朝地标中心拍。
   - 最后按 1.6 m 的拍摄高度确定 z。
   - 输出 ENU 坐标（x=东，y=北，z=上，单位米），以及每张照片估计出的位置和朝向。也就是说，照片的方位由 SfM 自动判断，不只依赖手机的 EXIF。
   - GPS 不够用时，`georegistered` 为 false。不够用的情况包括：内点太少、分布太窄、预计尺度误差太大，或者和罗盘偏航相差超过 40°。这时偏航只靠罗盘或伪罗盘，尺度是粗估的，在真实建筑上常偏大 1.5–2.4 倍。质量门不检查这一项，所以这种模型也可能发布。
   - **地理锚点**（`anchor.py`）：配准结果另存成一条锚点记录，和 SfM 自己的局部坐标分开：从局部坐标到 ENU 的相似变换，以及偏航、尺度、水平位置、高度、倾斜各自的 1σ 和来源；每台相机也另记局部位姿。σ 按平方和加上一组下限（`dev-core5-v1`，只在开发集上拟合）。锚点只是把已有的配准记下来，不带来新的偏航或尺度来源，所以模型的绝对位置没有变准。风格化的吸附门和 API 的 `geo` 块读它；质量门只记录、不拦。
   - 配准之后，`camera_flags` 逐台检查相机。离自己的 GPS 超过内点阈值（至少 10 m）、并且和自己的罗盘相差超过 45° 的，标为"双胞胎"，常见于长得像的两个立面。有 2 台以上时，重建会切断它们和其余照片的匹配，再建一次图。
8. **发布质量门**（`qa.py`）：只和产品本来就知道的东西比较（照片的 GPS 和罗盘、地标中心和半径），不需要真值。检查项和门槛：

   | 检查 | 门槛 |
   |---|---|
   | 注册相机 | 至少 3 台；注册率立面 ≥ 60%，全景 ≥ 70% |
   | 点数 | ≥ 300 |
   | 平均重投影误差 | ≤ 2.5 px |
   | 双胞胎相机 | ≤ 10% |
   | 带可疑标记的相机 | ≤ 20% |
   | 各相机罗盘偏差的中位数 | ≤ 25° |
   | 点云水平中位点离地标中心 | ≤ 半径的一半 |
   | 全景模型的覆盖 | 就绪扇区数和全景作业的要求相同，每个扇区至少 2 台未标记的注册相机 |

   没过门时：
   - PLY 和 `.json` 照样写盘，`.json` 里 `published` 为 false；
   - 作业记为失败，并写明原因；
   - 之前发布过的版本继续显示；没有的话，这个目标暂时没有模型。

   过门模型里被标记的相机，其位姿不公开。
9. **输出**：
   - `models/<地标>/<模型>.ply` 是带颜色的点云。
   - 同名 `.json` 里有注册数、点数、GPS 残差、朝向误差、每个相机的位姿（ENU 和局部坐标）、是否发布（`published`）、质量门结果、各阶段耗时、地理锚点（`anchor`）和带版本的输入（`inputs`：照片数、其中最新的上传时间、照片集的 sha256）。
   - 发布的 COLMAP 模型另把稀疏模型和 `map.json` 存进 `maps/<地标>/models/<模型>/`，每个目标留最新的 2 个。风格化作业读它。

### 风格化模型（结构优先；作业系统、脚本和基准共用一个入口）

风格化的入口有三个，都调用 `crowd3d.structure_pipeline`：
- **作业系统**（`jobs.py`）：地标有分件快照时，每个发布的 COLMAP 模型排一个风格化作业。
  - 重建作业优先；没有重建作业排队时，才开始风格化作业。
  - 作业在子进程里跑（`python -m crowd3d.structure_pipeline run --spec`），2 个线程，`stylize_timeout_s`（1200 s）一到就杀掉。
  - 子进程读模型存下的地图和照片的处理图，只写自己的工作目录和照片缓存，不碰数据库。
  - 风格化模型过了自己的质检，才写出 GLB。服务把它放到 PLY 旁边，记进 `ModelRecord.stylized`。
  - 目标已经显示更新的模型时，作业以说明结束。子进程失败或超时，作业失败，不写 GLB。
- `scripts/stylize_model.py`：处理单个模型。
- `scripts/benchmark.py`：把测试集上每个发布的模型都风格化一次（基准里产品自己的风格化作业是关的）。

**结构流水线做什么**（[docs/STRUCTURE_FIRST.md](docs/STRUCTURE_FIRST.md) §2.9）：读分件快照并核对 sha256 → 用 `stylize.register_scene` 吸附、过配准门 → 配准被接受、并且分件不超过 60 个时才跑判定器（D15）→ D25 只在有两种可商用天空掩码时跑（今天不跑）→ 发布门检查 → 风格化。没有分件快照时，退回外轮廓拉伸。

**只有一种天空掩码（A+R）时**：天空轮廓线这一路既不能核验也不能投票；D15 不采用整栋外框；D25 不跑。D15 只剩一种组合：SfM 点和三维直线一致，并且读到这块积木自己的屋顶。今天每个真实众包模型都过不了配准门，判定器根本不跑，所以产品给出的都是 OSM 分件的素模。

下面第 1–8 步说明整套设计。在结构流水线里，判定（第 3 步）和 D25（第 5 步）只在配准被接受时才跑。

1. **框架**（`structure.py`）。读地标的 OSM 快照，保留落在主轮廓里的 `building:part`。每块变成"底高到檐口的棱柱 + 屋顶原语"。
   - 高度取自 `height`、`building:levels`、`roof:height` 等标签。
   - 标签缺了，就用屋顶形状的约定比例（D17），或者 3 m 的"高度待定"矮台（D13）。
   - 每个参数都记下来源和状态（体块登记表 v2）。Wikidata 的总高永远不用于单个分件。
   - 快照在 `eval/osm/<slug>.parts.osm`，由 `scripts/trim_osm_parts.py` 从整块下载裁出。
2. **照片证据**。三路共用同一套位姿：
   - 天空轮廓线（`skyline.py`，基于 A+R 天空掩码）；
   - SfM 点；
   - 三维直线（`lines3d.py`）。
3. **判定**（`judge.py`，D15）。下面几个条件都满足，才自动采用照片值：
   - 至少两路给出数值，而且所有给出数值的通道两两相差 ≤ max(1 m, 5%)；
   - 方向相同；
   - 没有顶到搜索边界；
   - 共同尺度没确立时，还要通过尺度无关检查；
   - 父件的改动已被接受；
   - 改高度时（往上、往下都算），一致的证据里还要有一路读到这块积木自己的屋顶（D27）：三维直线的屋脊或顶点，或者这块积木的屋面拟合（`roof_fit.py`，和 D25 同一个拟合，而且离轮廓 ≥ 2 m 的格子里要有点，看得到屋顶本身）。只有天空轮廓线加三维直线的檐口、或者天空轮廓线加 SfM 点的最高处（0.98 分位），都不够，开工单。

   细高构件，以及坐标工单还没结的时候，一律不自动采用。条件不满足时保留地图值，并开一张工单。
4. **立面凹凸**（`relief.py`）。用照片的点量出壁柱、檐带、凹进的开间等。
   - 有 G-R1…G-R7 七道守卫，例如 G-R7 要求量到的边离杆子、树干这类独立物体 ≥ 0.3 m；还有"合并开间"检查。
   - 有三维直线核对的元素照常建。
   - 只凭点建的元素（三维直线只在偶然水平的，即 D24 元素），要求模型自己的数据能确立共同尺度；不满足时不建这类元素。
5. **强证据纠正 OSM**（`correct.py`、`correct_measure.py`、`tilt.py`，D25）。纠正三类值：
   - 分件高度和屋顶高，只往上改（D25-H）；
   - 塔的倾斜（D25-T）；
   - "高度待定"矮台和空隙的体块（D25-B）。

   前提是证据级的配准盒子和尺度无关的参照，并且一致的证据里要有一路读到这块积木自己的屋顶（屋面拟合，或三维直线的屋脊、顶点）。盒子太宽时只采用下界。D25 要两种天空掩码，所以产品里不跑。

   每处纠正都带地图原值和全部数字，可以逐条撤销。另外输出一份建议的 OSM 修正报告；OSM 本身不改。
6. **吸附、配准门和上色**（`stylize.py`）。
   - 模型先按窗口搜索吸附到 OSM 轮廓上。偏航窗口按模型报的偏航 σ 定；模型有地理锚点时用锚点的 σ。
   - 然后过配准门。门本身没有参考位姿，量不了真实误差，它检查的是：
     - 吸附后墙点到轮廓的中位距离 ≤ 1.5 m；
     - 逐边核验：有票的边里 ≥ 60% 的中位距离 < 3 m，通过的边方向至少差 30°；
     - 偏航别名：另一个差不多一样贴合的摆放，如果会把墙上的照片或相机挪动 ≥ 1.5 m，就拒绝；
     - 任务 #73 加的三条（流水线模型才查）：窗口被建筑的对称周期截短、或者模型没报偏航 σ 时拒绝；吸附后相机到自己 EXIF GPS 的中位距离超过 max(20 m, 3 × 锚点在相机处的水平 σ) 时拒绝；
     - 另外还检查 GPS 深度、相机位移、倾斜和可见性。

     基准事后拿参考位姿判"对/错"：相机 ≤ 1.5 m、三维墙点 ≤ 1.5 m 算对。
   - 过门才给墙上色。只用拍到的像素，量化到固定的调色板；墙上的小洞可以用这面墙的实测颜色补（D12）；没拍到的地方是素模。
   - 不过门时整栋都是素模。判定器从同一批照片得出的结论（D15 采用、照片核验）也全部撤回。D25 的纠正则按配准等级保留，所以素模上也可能带着照片纠正。
7. **诚实检查**。三道检查各管一层：
   - `check_honesty` 查流程本身执行的约束；
   - `audit_model` 用点和相机独立复核屋面和墙面的点支撑、遮挡和补洞，并核对 D24 的凹凸几何、D25 的每个纠正值都和各自的记录一致；
   - `audit_glb` 只看发布出去的 GLB 字节，核对每个节点的来源、每个照片值的证据和记录、纹素来源图集。

   这些检查用来抓程序 bug：它们核对记录完整、和模型一致，但不用记录里的数字把判定规则重算一遍，所以挡不住自洽地伪造的记录（D26）。

   另外，`structmetrics.py` 统计每个数字来自 OSM、约定、占位值还是照片。
8. **查看**。GLB 每个节点的 extras 都带来源和证据。在 `web/stylized.html` 里点选一块积木，可以看到它的来源卡片，例如"形状：OpenStreetMap 志愿者测绘"或"照片纠正地图（自动）"。
   - 画风文件在 `web/themes/*.json`。
   - 换画风时，查看器按颜色表逐色替换，不用重跑风格化。
   - 画风有校验规则，例如素模色和实拍色要分得开（ΔE00 ≥ 7.5），见 [docs/ART_DIRECTION.md](docs/ART_DIRECTION.md)。

## HTTP API

所有接口都是 JSON。时间是 unix 秒，角度是度，距离是米。字段的完整定义见 `crowd3d/api.py`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | `{"ok": true, "reconstructor": "colmap"}` |
| GET | `/api/landmarks` | `{"landmarks": [LandmarkSummary]}`：进度、扇区状态、最新全景模型 |
| POST | `/api/landmarks` | `{id?, name, lat, lon, radius_m?}` → 201 LandmarkSummary；id 重复返回 409，参数非法返回 422。不给 id 时由名称生成 slug，中文名生成 `lm-xxxxxxxx` |
| GET | `/api/landmarks/{id}` | LandmarkDetail：在 Summary 的基础上加共同完成者、每个类型/扇区的最新模型、最近 20 个重建作业（风格化作业不在列表里）、最近 200 张照片（含估计的位置和朝向）、阈值；不存在返回 404 |
| POST | `/api/photos` | multipart：`files`（1–50 个）、`user_id`、`landmark_id`（可选，不给时按 GPS 自动匹配地标）→ `{"results": [UploadResult], "triggered_jobs": [JobInfo]}`。照片本身的校验失败放在每个文件的结果里，不作为 HTTP 错误返回。请求本身出错时仍返回 HTTP 错误：`user_id` 非法或文件数不在 1–50 返回 422，`landmark_id` 不存在返回 404，请求体超过上限返回 413 |
| POST | `/api/landmarks/{id}/rebuild` | `{kind: "face"\|"full", sector?}` → 202 JobInfo；照片数没达到最低要求返回 409，kind 或 sector 非法返回 422 |
| GET | `/api/jobs/{job_id}` | JobInfo（kind 取值：face / full / stylize；status 取值：queued / running / done / failed；风格化作业的 `model_id` 是它处理的模型） |
| GET | `/files/models/<地标>/<模型>.ply` | 点云文件。风格化作业生成的 `.glb` 也在这里。旁边的 `.json`、`.style.json` 不提供（里面有每张照片的位姿） |
| GET | `/`、`/landmark.html?id=…` | 网页（`web/` 目录） |
| GET | `/stylized.html?model=<GLB 地址>&theme=<画风>` | 风格化查看器（静态页面）。`model` 只接受同源地址或相对路径，跨域地址会被忽略 |

ModelInfo（Summary 的最新全景模型、Detail 的模型列表）有两块和风格化、地理位置有关：
- `stylized`：
  - `status` 为 `ready` 时，`url` 指向 `/files/models/…/<模型>.glb`，`registration` 是 `{accepted, reason}`，`label` 是"风格化（照片已对齐）"或"素模（照片未对齐）"；
  - `status` 为 `pending` 时，风格化作业正在排队或运行；
  - 没有、也不会有风格化模型时为 null。
- `geo`：按现在的下限算出的地理锚点。
  - 字段：`level`（full / prior_only / none）、各分量的 1σ `sigma`、来源 `source`、`calibration_id`、点中位处的水平和竖直 σ `position_sigma_m`、`calibrated`；
  - `calibrated` 只在标定 id 列在 `Settings.calibrated_anchor_ids` 里时为 true。这个列表现在是空的，所以地标页不显示"位置 ±X 米（1σ）"；
  - 锚点出现之前的模型为 null。只供显示，质量门不读它。

结构框架本身没有 API，也没有"重新风格化"的接口。

## 项目结构

```
crowd3d/
  config.py           Settings：产品阈值（照片数、校验、发布质量门）、数据目录、产品里的风格化
  params.py           算法内部参数（重建、特征缓存、续建、地理锚点下限、校验、质量门），可以整套替换，基准用它比较参数组
  geo.py              ENU 局部坐标、方位、距离
  exif.py             读写 EXIF（GPS、指南针、焦距、机型）
  records.py          Landmark / PhotoRecord / Job / ModelRecord
  validation.py       照片校验（拒绝码、警告码、模糊度、dHash）
  coverage.py         扇区、覆盖状态、作业规划（纯函数）
  store.py            SQLite 存储（线程安全）
  jobs.py             JobRunner：后台作业线程（重建作业、风格化作业）、带版本的输入、唯一的替换规则
  ingest.py           单个上传：校验 → 入库 → 保存原图和处理图
  reconstruct.py      ColmapReconstructor、地理配准、双胞胎检测、FakeReconstructor
  anchor.py           地理锚点 GeoAnchor：局部坐标到 ENU 的相似变换，每个分量的 σ 和来源
  landmark_map.py     地标的持久地图：特征和匹配缓存、已发布模型的稀疏地图、续建的底模（默认关）
  qa.py               发布质量门（地理锚点的检查只记录）
  synthetic.py        光线投射渲染的合成建筑和照片（含 6 类坏照片）
  api.py              FastAPI 应用和命令行入口
  photoposes.py       每张照片的拍摄位置、朝向和误差范围（目前只给基准的 photo_pose_* 指标用）
  structure_pipeline.py  结构优先的唯一入口（命令行、基准、风格化作业的子进程）；只认 A+R 天空掩码
  scene_io.py         读作业或已存地图的稀疏模型和模型 JSON（按锚点放到 ENU）、吸附先验、点的轨迹、轮廓
  stylize.py          风格化：体块、吸附和配准门（含 #73 的三条）、上色、素模、D12 补洞、GLB、诚实检查
  structure.py        OSM building:part → 参数化积木（屋顶原语、来源登记表、无照片的结构检查）
  skyline.py          天空掩码（A+R）和天空轮廓线证据
  lines3d.py          三维直线证据
  judge.py            D15 判定器：两路一致自动采用（改高度要读到自己的屋顶，D27）、工单、配准被拒时撤回
  roof_fit.py         屋面拟合和它的可用条件（D25 和 D15 共用）
  relief.py           D24：照片实测的立面凹凸
  correct.py          D25：纠正规则、记录、撤销、发布门、OSM 修正建议
  correct_measure.py  D25 的测量：配准盒子、墙顶、屋顶读数
  tilt.py             D25-T 的倾斜通道
  structmetrics.py    结构指标：多少细节、每个数字来自哪里
web/
  index.html、landmark.html   地图首页、地标页（罗盘、上传、排行、作业、模型和风格化链接、三维点云查看器）
  index.js、landmark.js       两个页面的逻辑
  api.js                      接口调用和错误码的中文文案
  viewer.js、framing.js       three.js 点云查看器和取景
  compass.js                  覆盖罗盘
  status.js、models.js        作业状态、模型排序和风格化链接
  ui.js、style.css            公共组件和样式
  stylized.html、stylized.js  风格化 GLB 查看器（来源卡片、画风切换）
  themes.js、themes/          画风文件和画风校验
  tests/                      node 单元测试
scripts/
  demo.py                     分阶段上传的演示
  stylize_model.py            从一个作业或参考场景生成风格化 GLB（有分件快照时走结构流水线）
  benchmark.py                在冻结测试集上跑完整产品链路并记录指标（run / compare / report / rescore / posediff / calibration）
  eval_real_landmark.py       单栋建筑的评测：按众包方式上传，和参考答案比较；全部指标的定义
  fetch_commons_landmark.py   从 Wikimedia Commons 取一栋建筑的测试照片（只要 EXIF 自带 GPS 的）
  materialize_testset.py      按冻结清单把测试集照片取到仓库外，逐张核对 sha256
  trim_osm_parts.py           从整块 OSM 下载裁出分件快照（裁后建出的结构和整块完全相同才写）
  replay_map_views.py         持久地图第 2a 步的离线重放：从一张图切出扇区视图（结论是不做 2b）
  export_for_splatting.py     把 COLMAP 模型导出给 3D 高斯泼溅训练工具
eval/
  landmarks.json              地标目录：坐标、OSM 轮廓、分件快照（osm_parts）、Wikidata 事实
  osm/                        OSM 外轮廓（GeoJSON）和裁剪后的分件快照（<slug>.parts.osm，13 个地标），ODbL
  testsets/                   冻结的测试集清单（只有元数据）
  splits/                     开发 / 留出划分（commons-ten-v1-holdout-v1.json）
  criteria/                   compare --criteria 的通过标准（review-v1.json）
  configs/                    基准的参数组（default、fast、cache、continue）
  results/                    每次运行的数字结果和 LEADERBOARD.md；B0、F 和 ablation-v5 拆分格子
tests/                pytest；test_e2e.py 是端到端验收、test_e2e_stylize.py 是风格化作业端到端（都是 slow），test_web.py 调用 node 测试
docs/                 ARCHITECTURE、STYLIZED_MODEL、STRUCTURE_FIRST、ART_DIRECTION、BENCHMARK、REAL_WORLD_EVAL、
                      REVIEW_ADOPTION（外部评审的采纳和最终评测）、VISUAL_LOCALIZATION（视觉定位，只有设计）、
                      PHOTOREAL_LOCAL（本地训练 3DGS）
```

数据目录（`Settings.data_dir`，默认是 `./data`）：

```
photos/<地标>/<照片>.jpg       原图（只保存通过校验的照片）
photos/<地标>/<照片>.proc.jpg  处理图：已按 EXIF 方向转正，长边 ≤ 1600，EXIF 只保留焦距和机型
models/<地标>/<模型>.ply/.json 模型和元数据
models/<地标>/<模型>.glb       风格化作业的 GLB（旁边的 .style.json 是它的摘要，不对外提供）
maps/<地标>/cache.sqlite3      特征和匹配缓存（不淘汰，每张照片约 1.1–1.7 MB）
maps/<地标>/models/<模型>/     已发布 COLMAP 模型的稀疏模型和 map.json（每个目标留最新 2 个）
maps/<地标>/structure-cache/   风格化作业的照片缓存（天空掩码、线段）
work/<作业>/                   COLMAP 工作目录；风格化作业用 work/<作业>.<第几次>/
crowd3d.sqlite3
```

不合格的上传只在数据库里留下记录（原因、位置），不保存图片。文件名一律用 uuid，不使用客户端传来的文件名。`maps/` 下的东西都不对外提供。

## 配置

产品阈值在 `crowd3d/config.py` 的 `Settings` 里。命令行只开放了数据目录和重建器两项，其余字段在代码里构造 `Settings(...)` 时修改。环境变量 `CROWD3D_STRUCTURE_CATALOG` 可以换一个地标目录，设成空值就关掉风格化作业。

其他参数：
- `crowd3d/params.py` 的 `Params`：算法内部参数，例如 GPS 和罗盘的 σ、RANSAC 门槛。和这一轮有关的几项：
  - `feature_cache`（默认 True）：地标特征缓存；
  - `map_continue`（默认 False）：从目标上一个模型续建，开发集验证没过；`map_rebuild_ratio`（1.5）、`map_max_continuations`（4）等是它的规则；
  - 地理锚点的下限：`anchor_yaw_floor_deg` 7.35°、`anchor_scale_floor` 0.352（log 尺度）、`anchor_xy_floor_m` 3.1 m、`anchor_tilt_floor_deg` 1.94°，高度 σ `anchor_z_sigma_m` 1.5 m，这组下限叫 `anchor_calibration_id` = `dev-core5-v1`。
- 基准的 `eval/configs/*.json` 只写要改的字段，没写的保持默认值。可以改的有：重建器、`Params`，以及 `Settings` 的产品阈值。不能改的是数据放在哪、进程怎么跑、产品自己的风格化步骤和只供显示的字段：`data_dir`、`reconstructor`、`inline_jobs`、`keep_work_dirs`、`max_job_attempts`、`stylize_published`、`structure_catalog`、`structure_bindings`、`stylize_timeout_s`、`calibrated_anchor_ids`。
- 风格化和结构优先各模块的参数，都在代码里：
  - `crowd3d/stylize.py` 的 `StyleParams`（默认 `DEFAULT_PARAMS`），包括 #73 的三个开关 `snap_use_anchor`、`snap_refuse_clipped_window`、`snap_max_gps_median_m`（20 m），默认都开；
  - `crowd3d/structure_pipeline.py` 的 `Limits`（每个模型 1200 s、4096 MB、最多判 60 个分件，D25 `auto`，掩码只有 A+R）；
  - `crowd3d/structure.py` 的 `StructureParams`（`DEFAULT_STRUCTURE`）；
  - `crowd3d/relief.py` 的 `ReliefParams`（`DEFAULT_RELIEF`）；
  - `crowd3d/lines3d.py` 的 `LinesParams`（`DEFAULT_LINES`）。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `data_dir` | `$CROWD3D_DATA_DIR` 或 `data` | 数据目录 |
| `num_sectors` | 8 | 扇区数，0 号以正北为中心，顺时针编号 |
| `face_min_photos` | 4 | 一个扇区有多少张合格照片才生成立面模型 |
| `face_rebuild_growth` | 3 | 立面模型每多几张照片重建一次 |
| `face_margin_deg` | 10 | 立面作业额外使用相邻扇区多少度以内的照片 |
| `full_min_photos` / `full_min_sectors` | 24 / 6 | 生成全景模型需要的照片总数和就绪扇区数 |
| `full_reachability_min_photos` | 48 | 照片达到这个数以后，仍然没有照片的方向算作"拍不到"，不再等它 |
| `full_rebuild_growth` | 8 | 全景模型每多几张照片重建一次 |
| `default_radius_m` | 150 | 新建地标的默认打卡半径 |
| `max_heading_offset_deg` | 60 | 指南针朝向偏离地标方向的上限 |
| `min_image_side_px` | 480 | 照片短边的最小像素数 |
| `blur_min_laplacian_var` | 30 | 模糊度阈值（在长边 ≤ 800 px 的灰度图上计算） |
| `duplicate_max_hamming` | 4 | dHash 距离不超过此值即判为重复 |
| `max_upload_bytes` | 25 MB | 单个文件的大小上限 |
| `max_image_pixels` | 5000 万 | PNG / WebP 的像素上限，超过判为 `too_large`（这两种格式按原尺寸解码，太大会占满内存）。JPEG 不受这项限制 |
| `processing_max_side_px` | 1600 | 送入 SfM 的处理图长边上限 |
| `reconstructor` | `$CROWD3D_RECONSTRUCTOR` 或 `colmap` | `colmap` 或 `fake` |
| `camera_height_m` | 1.6 | 假定的拍摄高度（手机的 GPS 高度太不准，不用） |
| `inline_jobs` | False | 在请求内同步执行作业（测试用） |
| `keep_work_dirs` | False | 保留每个作业的 COLMAP 数据库和模型（每张照片约 1.5 MB）；风格化脚本的 `--job-dir` 需要 |
| `max_job_attempts` | 2 | 同一个作业最多尝试几次：进程崩溃中断它这么多次后，判为失败，不再重跑 |
| `stylize_published` | True | 发布的模型自动排风格化作业（地标要有分件快照） |
| `structure_catalog` | `eval/landmarks.json`（`$CROWD3D_STRUCTURE_CATALOG`，空值为关） | 带分件快照的地标目录 |
| `structure_bindings` | {} | 地标 id → 目录 slug，给 id 不等于 slug 的地标用（目录中心仍要在地标半径内） |
| `stylize_timeout_s` | 1200 | 风格化子进程的时间上限，超时就杀掉，这个模型没有 GLB |
| `calibrated_anchor_ids` | ()（空，关） | 通过留出集标定的锚点标定 id；只有列在这里，`ModelInfo.geo.calibrated` 才为 true、地标页才显示"位置 ±X 米（1σ）" |
| `qa_min_registered_fraction` / `qa_full_min_registered_fraction` | 0.6 / 0.7 | 发布质量门：立面 / 全景模型的最低注册率 |
| `qa_min_points` | 300 | 发布质量门：最少点数 |
| `qa_max_reprojection_error_px` | 2.5 | 发布质量门：平均重投影误差的上限 |
| `qa_max_twin_fraction` | 0.1 | 发布质量门：双胞胎相机（GPS 和罗盘都对不上）的比例上限 |
| `qa_max_flagged_fraction` | 0.2 | 发布质量门：带任何可疑标记的相机比例上限 |
| `qa_max_compass_error_deg` | 25 | 发布质量门：各相机与罗盘偏差的中位数上限 |

## 测试

分四层：合成数据上的自动化测试（每张照片的真实位姿都已知）、真实 COLMAP 的端到端验收、Web 的 node 测试，以及在冻结测试集上跑的基准（含 10 栋著名建筑的真实照片）。

- `python -m pytest -q -m "not slow"`：928 个快速测试，都不需要真实照片。覆盖范围：
  - 点云链路：地理计算、EXIF、校验、规划、存储、作业（含风格化作业和替换规则）、上传、API、发布质量门（`qa.py`）、合成数据。
  - 这一轮新加的：地理锚点（`test_anchor.py`）、特征缓存和已存地图（`test_landmark_map.py`）、重建输出的金样测试（`test_recon_golden.py`，固定 `RECON_OUTPUT_VERSION`）、#73 吸附门（`test_snap_gate73.py`）、结构流水线（`test_structure_pipeline.py`）、照片位姿（`test_photoposes.py`）、分件快照裁剪（`test_trim_osm.py`）、扇区视图重放（`test_replay_map_views.py`）。
  - 风格化：配准门和诚实检查（`stylize.py`）。
  - 结构优先：OSM 分件和屋顶原语（`test_structure.py`、`test_structure_stylize.py`）、天空轮廓线和 D15 判定（`test_skyline_judge.py`、`test_d15.py`）、三维直线、D24 立面凹凸（`test_relief.py`、`test_d24.py`）、D25 纠正（`test_d25.py`、`test_d25_measure.py`、`test_d25_publish.py`），以及结构指标。
  - 基准和测试集脚本的纯函数（`test_benchmark.py`、`test_eval_real.py`、`test_materialize_testset.py`、`test_fetch_commons.py`）。
  - Web 的 node 测试。
- `python -m pytest -q -m slow`：22 个慢测试，大多跑真实 COLMAP：
  - 重建：单个立面、全景、空间匹配、焦距先验、照片顺序不影响结果、用特征缓存的作业只算新的、续建（开关打开时）。
  - 特征缓存：COLMAP 跳过写进去的特征和图像对，算出的其余部分相同。
  - `tests/test_e2e_stylize.py`：通过 HTTP API 上传合成方盒的两面，两个立面作业先重建，然后每个发布的模型在子进程里风格化，地标页的模型带着服务提供的 GLB 和配准标签。
  - #73：记录下的 69 个对抗错误通过，开关打开后全部被拒。
  - `tests/test_e2e.py`。端到端测试用 640x480 的合成照片，8 面各 4 张，外加 6 张坏照片，通过 HTTP API 上传，按以下门槛验收：
    - 先上传北面，只触发一个立面作业，得到已地理配准的立面模型；
    - 再上传其余照片，全景模型要满足：注册率 ≥ 80%，已地理配准，GPS 残差 < 5 m，朝向误差 < 10°，≥ 80% 的相机离真值 < 5 m，≥ 1000 个点，点云水平中位数离原点 < 6 m；
    - 坏照片逐一被拒，原因与预期一致；
    - 这些门槛只适用于合成数据的小噪声（GPS σ 2 m、罗盘 σ 8°）。
- `node --test 'web/tests/*.test.mjs'`：95 个，94 个通过，1 个跳过。跳过的那个用 Khronos glTF 校验器检查风格化 GLB，要设 `GLTF_VALIDATOR` 和 `STYLIZED_GLB` 才跑。覆盖范围：罗盘、API 封装和中文文案、模型列表（风格化链接和标签、"GPS 对齐残差"）、作业轮询、三维取景、风格化查看器（来源卡片、D24 和 D25 的文案）、画风校验，以及用假 DOM 跑的首页和地标页。
- 最近一次完整运行：程序的核查者在提交 5e44646 上跑的。快速测试 928 个、慢测试 22 个全部通过；node 测试 94 个通过、1 个跳过。之后的提交只改了结果文件、文档、一处注释和 `tests/test_benchmark.py` 里的一个测试，测试数不变。
- **基准**（`scripts/benchmark.py`，不在 pytest 里，要先把照片取到仓库外）：同一批冻结的照片按真实用户的方式走完整条产品链路，记录发布、配准、风格化、结构和诚实检查的指标，并和参考答案、OSM 轮廓比较。用法、指标含义和噪声基线见 [docs/BENCHMARK.md](docs/BENCHMARK.md)，结果和结论见 [docs/REAL_WORLD_EVAL.md](docs/REAL_WORLD_EVAL.md) 和 [docs/REVIEW_ADOPTION.md](docs/REVIEW_ADOPTION.md)，全部运行的汇总见 [eval/results/LEADERBOARD.md](eval/results/LEADERBOARD.md)。
  - 三个测试集：`synthetic-v1`（合成，几分钟）、`mdx-two-v1`（早期两栋建筑的 49 张真实 GPS 照片）、`commons-ten-v1`（10 栋著名建筑的 1,045 张 Wikimedia Commons 照片，每张自己的 EXIF 都带 GPS）。
  - 开发 / 留出划分（`eval/splits/commons-ten-v1-holdout-v1.json`）：开发集是核心五栋（洗礼堂、亚琛、勃兰登堡门、比萨斜塔、圣母教堂）加上整个 synthetic-v1 和 mdx-two-v1，只在这里调参。留出集是另外五栋（圣家堂、匈牙利国会大厦、林肯纪念堂、印度门、悉尼歌剧院），只用事先写明的冻结提交跑：基线 B0 和候选 F 各两次重复。
  - 指标第 7 版加了地图上的位置误差、轮廓的支持覆盖（主要指标是建筑局部 1.5 m 的 `cov_local_supported_m_1p5`）、锚点 σ 的标定、旧输入盖掉新输入的次数和运行代价。`compare --criteria review-v1` 按通过标准判 PASS / FAIL。
  - 最终评测（F 对 B0）：硬守卫在两个划分上都是 0；dev 是 FAIL（主要指标 134.85 → 136.8 m，没超出噪声，而 dev 要求显著变好）；test 是 INSUFFICIENT（有数据的规则都通过，但三栋没有发布的模型）。作业 CPU 秒核心五栋少 59.7%。
  - 风格化配准门：修复前放过了 51 个模型里的 31 个（0.608），可核对的 28 个里有 19 个其实放错了；修复后一个都不过。F 上可参评的真实模型 dev 58 个、test 6 个，也一个都没过。
  - 基准经结构流水线风格化，结果里有 `struct_*` 指标。真实模型上 `struct_checked`、`struct_d15_adopted`、`struct_photo_share` 都是 0，因为配准都被拒、判定器不跑；结构的守卫也都是 0。早期的结构优先参考结果（洗礼堂、亚琛、比萨）记在 [docs/STRUCTURE_FIRST.md](docs/STRUCTURE_FIRST.md)。
  - 大约每 7 张 Commons 照片只有 1 张带 GPS（各建筑中位数 15.3%）；带 GPS 的照片里约 2/5 还带罗盘。
  - 照片各有许可，只有元数据（标题、作者、许可、坐标、哈希、数字结果）进仓库；照片、点云、GLB 和截图都不进仓库。OSM 分件快照（ODbL）进仓库。

## 原型的局限

- **真实照片只测了十栋建筑，阈值是样本内的。** 发布质量门、风格化配准门、#73 的开关和锚点下限都是在开发集（核心五栋、合成集、mdx-two-v1）上定的。留出五栋只用冻结的提交跑过 B0 和 F，其中三栋（林肯纪念堂、印度门、悉尼歌剧院）一个模型都不发布。所以留出集上的通过标准和标定都是"证据不足"。见 [docs/REVIEW_ADOPTION.md](docs/REVIEW_ADOPTION.md)。
- **真实照片的风格化模型目前全是素模。** 配准门要求相机位置误差 ≤ 1.5 m；拍摄者离墙 40–90 m 时，只靠 OSM 轮廓和手机 GPS 证明不了这么准，所以真实建筑一面墙都不上色。八边形这类轮廓（洗礼堂）在锚点的偏航 σ 下，窗口会被对称周期截短，众包模型的吸附都会被拒。要恢复上色，需要不靠轮廓的偏航和尺度来源（[docs/VISUAL_LOCALIZATION.md](docs/VISUAL_LOCALIZATION.md) 的视觉定位，只有设计），或者产品决定放宽标准。
- **模型在地图上的位置不确定度现在报出来了，而且很宽。** API 的 `geo` 块给出每个分量的 σ。尺度 σ 的中位数约 ±43%；在核心五栋上，点中位处的水平 σ 中位数是 16.9 m（7.7–47.1 m），大部分来自尺度。留出集上标定证据不足，匈牙利国会大厦的水平 σ 还说小了（6 个模型里 4 个在 3σ 外），所以地标页的"位置 ±X 米（1σ）"关着。锚点也没让位置变准：地理相机误差的中位数 dev 6.58 m、test 15.07 m，评审之前是 5.99 m、15.78 m。
- **配准门的已知漏洞（任务 #73）已经修了，但"0 个错误通过"仍只对已记录的运行和对抗用例成立。**
  - 现在门读锚点的偏航 σ；窗口被对称周期截短、或者模型没报偏航 σ 时拒绝；吸附后相机离自己 EXIF GPS 的中位距离太远（> max(20 m, 3 × 锚点的水平 σ)）时拒绝。对抗用例里错误通过从 69 个降到 0 个。代价是精确别名那组里本来对的 78 个通过只剩 16 个。
  - `scripts/stylize_model.py --snap-from` 复用旧吸附前，流水线模型要先按现在的门重新核对。
  - 几处决定上色的检查还没有回归测试。真实模型没有一个走到门的"通过"分支，所以开关的好处在真实数据上看不出来。

  D25 也依赖这一点：近圆形的轮廓（洗礼堂有 23 个对称别名）上，纠正要在所有别名转角下都成立才采用。
- **框架几乎全部来自 OSM，照片的贡献很小。** 结构优先让模型认得出来了，例如亚琛的分件、穹顶和尖顶，但形状主要是 OSM 志愿者画的。照片的贡献：
  - 洗礼堂：打开 D24 以后，85 个数字里有 36 个来自照片，大多是立面凹凸。
  - 亚琛：D24 的尺度条件不满足，没有打开；D25 只纠正了一处，唱诗堂顶高"至少 36.26 m"（OSM 写 32 m），照片数字从 0% 升到 0.21%。
  - 比萨：D25-T 只在参考模型上通过，量出倾斜 4.59°，朝 177.0°（公开值约 3.97°）。已发布的众包模型都不带倾斜：它们只拍到塔的一面，点云轴不够准。
  - 洗礼堂上 D25 没有改任何值。
  - 以上数字都来自参考场景（全部照片、已知位姿），而且另用了研究用的第二种天空掩码 Dm+R。只用产品的 A+R 时，洗礼堂和亚琛的参考场景上照片数字都是 0，D25 不跑。众包等级上的 D25-H 和 D25-B 本来也关着。

  OSM 没画分件的建筑，只能把外轮廓拉伸成体块；标签缺了的地方，用约定比例或 3 m 的"高度待定"矮台代替。只在 Wikidata 有条目、地图上没有轮廓的地标还不支持。
- **结构优先接进了产品，但只给出 OSM 分件的素模。**
  - 产品给的：有分件快照的地标（目录里 13 个），每个发布的 COLMAP 模型一个 GLB，带 OSM 分件的体块、屋顶原语和来源，标"素模（照片未对齐）"。风格化作业的子进程 CPU 中位数 4.9 s、最多 19.7 s，峰值内存不超过 300 MB（核心五栋的 20 个模型）。
  - 产品不给的：照片颜色和照片量出的数字。原因有两个：每个真实众包模型都过不了配准门，判定器不跑；只有一种可商用的天空掩码，按"两种掩码"的规则，D25 不跑，天空轮廓线一路不起作用。
  - 没有分件快照的地标不出 GLB。
  - 没有"重新风格化"的入口：这个功能上线以前发布的模型，以及分件快照或风格化代码改了以后的模型，要等下一次发布才有新的 GLB。
- **D24 和 D25 按放宽后的标准运行（D26，2026-10-28）。** 对抗加固做了五轮也没有收敛，产品负责人决定放宽标准：
  - 只为合成攻击场景或伪造记录加的守卫都删掉了，D25 也不再往下改高度。真实数据的结果和清理前逐项相同，发布出去的 GLB 二进制部分逐字节相同。
  - 已知能骗过 D25 的合成情形：树冠盖满平顶、脚手架网围住"高度待定"矮台、网加临时顶棚的整体围护。它们会发布比真值高的值或下界。完整列表见 [docs/STRUCTURE_FIRST.md](docs/STRUCTURE_FIRST.md) §2.4.8 第十五部分。
  - 发布检查不防伪造记录。
  - D15 改高度现在要读到这块积木自己的屋顶（D27，2026-10-28）。合成场景里，离墙很近的脚手架网顶、女儿墙不再被当成檐口采用；代价是屋顶从地面看不见的平顶分件，以及小于约 8 m × 8 m 的分件（除非三维直线量到屋脊或顶点），D15 不再改。树冠盖在屋顶上时，树冠上的点能拟合出一个"屋面"，D15 仍可能采用树冠的高度。
- **全景模型在真实建筑上一次都没发布过**，每栋却要花 15–24 分钟反复尝试。不设上限跑完的勃兰登堡门和圣母教堂也没发布。游客只拍"好看的那一面"，很多方向几乎没有照片。
- **采集端**：
  - 只收相机原图：没有 EXIF 判 `no_exif`，EXIF 里没有 GPS 判 `no_gps`。截图、修图软件或聊天软件转存的副本，通常两样都没有。
  - 手机浏览器从相册选图时，可能会去掉 GPS，Android 上很常见。Commons 上也只有约 1/7 的照片带 GPS。
  - 不支持 HEIC，因为没有安装 HEIF 插件。
  - 磁北朝向只给出警告，没有做磁偏角修正。
- **几何**：
  - 重力方向的估计假设手机竖直拿、地面大致水平，单个立面可能残留几度的倾斜。勃兰登堡门的模型相对参考的中位倾斜是 3.4°，原因没查清。
  - 无法地理配准时，回退方案的尺度比较粗略。
  - 重建只输出稀疏点云，没有稠密重建和 3DGS。[docs/PHOTOREAL_LOCAL.md](docs/PHOTOREAL_LOCAL.md) 说明怎样导出数据，在本地用 GPU 训练 3DGS。
- **重建**：只用 CPU，单线程串行。特征和匹配现在有地标级缓存，但建图每次仍从零开始，没有增量注册：续建做了，可是开发集验证没过（版本连续性 56%，要求 90%；先试续建再从零重建，净省的 CPU 为零），默认关。用缓存时，不设上限跑完勃兰登堡门（150 张照片）要 1,345 s 墙钟，圣母教堂（111 张）要 3,953 s。
- **服务**：单机 SQLite 加本地文件，一个数据目录只能由一个服务进程使用。没有账号、鉴权、限流和内容审核，`user_id` 只是一个表单字段。单次请求最多 50 × 25 MB，超出时在写盘前返回 413。照片的估计位姿（已过质检、未被标记的）、模型点云和风格化 GLB 对所有人公开。进程崩溃中断的作业最多重跑 1 次，同一个作业被中断 2 次（`max_job_attempts`）后判为失败。`work/` 目录在作业结束后自动清理（`keep_work_dirs` 可保留）。`maps/` 下的特征缓存不淘汰，每张照片约 1.1–1.7 MB。
- **基准**：`scripts/benchmark.py` 的默认数据目录写死成开发机上的一个临时目录，在别的机器上要设 `CROWD3D_TESTSET_ROOT`、`CROWD3D_BENCH_DIR` 或每次传参数；新重建器要改代码注册；风格化的参数还不能通过配置调；逐照片指标还不按"没见过的摄影师"分开报。
- **网页**：依赖 CDN 和 OSM 公共瓦片。立面模型的相机叠加只画本扇区的照片。

## 下一步

外部评审采纳计划的结论和下一步见 [docs/REVIEW_ADOPTION.md](docs/REVIEW_ADOPTION.md) 第 5、6 节。

1. **视觉定位**（[docs/VISUAL_LOCALIZATION.md](docs/VISUAL_LOCALIZATION.md)，只有设计）：新照片对已配准的地标级模型做绝对定位，让没有 GPS 的照片加入模型，同时提供不依赖轮廓的偏航和尺度。这是唯一能改变绝对位置和配准门结果的路，也是真实照片恢复上色的前提。底子已经有了：每个已发布模型的稀疏地图存在 `data/maps/<地标>/models/` 下，锚点是独立的记录。
2. **单掩码的决定（D28）**：要么保留"两种掩码"的规则（现在的默认），要么让 A+R 单独算一路、门槛更严，后者要有自己的一轮验证。先要一个研究开关，把 A+R 自己的读数报出来才能权衡。另一条路是查清 Dm+R 或 C+R 的许可，得到第二种可商用的掩码。
3. **续建要一个更好的接受规则**：例如只在照片数超过 30 张时续建（小照片集保留三个种子的多样性），接受续建的门槛不用 `suspicious()` 的 5%，再定完整模型的重试规则（[docs/BENCHMARK.md](docs/BENCHMARK.md) 6.8 节末尾）。改完要重新过开发集验证。
4. **评审里需要 GPU 的阶段**：GPU 特征和匹配（如 ALIKED + LightGlue）、稠密重建或 3D Gaussian Splatting。这台机器只有 4 核 CPU，验证不了；到有 GPU 的机器上时，同一个划分和 `review-v1` 可以直接拿来比。
5. **留出证据太少**：五栋里三栋一个模型都不发布，凑不出标定要求的"至少 3 栋、每栋至少 3 个模型"。要加建筑，发一个新的测试集或划分版本，并用真实建筑做一次小规模实地采集，重新校准阈值，包括 GPS σ、模糊度和重复判定。
6. **引导拍摄**（方向 B）：采集端改为页面内拍摄，同时记录 GPS 精度、指南针和重力方向（DeviceOrientation），提示"还缺哪几面、哪几块积木还没被拍到"。
   - 这样可以绕开相册丢失 GPS 的问题，提高重力和朝向的先验。
   - 照片能给框架提供的证据也会更多。
7. **产品的缺口**：加"重新风格化"的入口；在现有的发布质量门之上，加"新版本模型不比旧版本差"的比较；完整模型一次都没发布过却占大量作业时间，要改它的重试规则。
8. **照片位姿的公开接口**（任务 #87）：基准部分已经合并（`crowd3d/photoposes.py`、`photo_pose_*` 指标）；在三维模型上画出每张照片的拍摄位置和朝向，要先定隐私规则。
9. **平台**：改用 Postgres + 对象存储，加作业队列和多个 worker，加账号、限流、人脸和车牌打码、隐私收紧，以及反作弊。

各阶段的路线图见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 第 10 节。
