# 外部评审的采纳：做了什么、测出了什么

> 2026-10-30。本文汇总 2026-10-29 开始的"外部评审采纳计划"：评审提了什么、采纳了哪些、为什么；按阶段做了哪些提交；怎么评测；结果；没变好的地方和下一步。
>
> **数字出处**：表里的数都来自结果文件，表下注明文件名。`eval/results/…` 在仓库里（只有数字和路径，没有照片、点云和 GLB）；比较、标定判定和吞吐量这几类派生结果在仓库外的开发机临时目录 `/tmp/claude-0/-home-user-temp/04ee4750-a5e3-5a4d-b130-9dfde89f5621/scratchpad/review/`（下文写作 `$R/`），第 7 节给了在仓库里重新算出它们的命令。重复两次的运行记两次的中位数，所以会有 x.5。

## 0. 先说结论

- **冻结的候选 F**（提交 5e44646，全部默认值：地标特征缓存开、续建关、任务 #73 的吸附门开关开、风格化走结构流水线、地理锚点下限 `dev-core5-v1`）和**基线 B0**（819cb78 的产品代码，也就是评审之前的产品）比：
  - **硬守卫在 dev 和 test 上都是 0**：错误通过、上错色的墙、风格化质检、结构判错、结构质检、旧输入盖掉新输入（test 上两个结构守卫只在有发布模型的两栋上有值）。
  - **dev 的通过标准是 FAIL**：主要指标（建筑局部 1.5 m 的支持覆盖）134.85 → 136.8 m，+1.95 m，没超过 9.9 m 的噪声门槛，而 dev 要求"显著变好"；另有一个守卫在比萨斜塔上被标出（地理 5 m 的错误覆盖 0.5 → 2.0 m），不用缓存的运行里也出现过同样的值，是 SfM 运行间的几种结果之一。
  - **test（留出五栋）的结论是 INSUFFICIENT**：有数据的规则全部通过（主要指标 33 → 34 m，不变差）；标定只有 4 个模型（要 20 个），三栋建筑一个模型都没发布，结构的守卫在它们上面没有值。
- **代价大幅下降**（作业 CPU 秒）：核心五栋 −59.7%（和同一请求前缀、不用缓存的对照比），留出五栋 −72%（基线当时机器较忙，这个数偏大），synthetic-v1 −32%，mdx-two-v1 −29%。请求延迟的 90 分位：dev 178.85 → 73.7 s，test 74.05 → 20.8 s。
- **留出集上的标定证据不足**：要求至少 3 栋留出建筑各有至少 3 个能核对的模型，实际只有匈牙利国会大厦一栋；而且那里水平位置的 σ 说小了（6 个模型里 4 个在 3σ 外）。所以地标页的"位置 ±X 米（1σ）"继续关着。
- **没变的**：模型在地图上的绝对位置（地理相机误差中位数 dev 5.99 → 6.58 m，test 15.78 → 15.07 m；地理 1.5 m 的支持覆盖在真实照片上一直是 0）；真实照片的模型仍然一个都过不了配准门，所以产品里的结构风格化给出的是 OSM 分件的素模，照片量出的数字是 0。

## 1. 评审提了什么，采纳了多少

评审的意见和采纳决定记在 2026-10-29 的决策记录（`/tmp/claude-0/-home-user-temp/04ee4750-a5e3-5a4d-b130-9dfde89f5621/scratchpad/user_decisions.md`）里，四个采纳包的设计和交叉核对在 `/tmp/claude-0/-home-user-temp/04ee4750-a5e3-5a4d-b130-9dfde89f5621/scratchpad/review_design.json`。

| 评审的意见 | 结论 | 为什么 | 做在哪里 |
|---|---|---|---|
| 地理配准和局部重建分开保存，每个分量给一个标定过的不确定度 | **部分采纳** | 锚点记录、σ 下限、读锚点的吸附门、API 的 `geo` 块都做了；"位置 ±X 米"没打开，因为留出集的标定证据不足、匈牙利国会大厦的水平位置 σ 说小了（4.5 节）。锚点只是把已有的配准记下来、给出 σ，不带来新的偏航或尺度来源，绝对位置不会因此变准 | `crowd3d/anchor.py`、`reconstruct.georegister`、`Params.anchor_*`、`stylize` 的 #73 开关、`api.model_info` 的 `geo` |
| 持久的地标地图：特征和匹配缓存、从上一个模型续建、输入带版本、旧输入不能盖掉新输入 | **部分采纳** | 缓存：建图读的数据库逐字节相同，作业 CPU 少一半以上，**默认开**。输入版本和替换规则、保存已发布模型的稀疏地图：**采纳**（`stale_overwrites` 一直是 0）。续建：做了，dev 验证没过（版本连续性 56%，要求 90%；净省 CPU 为零），**默认关**。一张图切出各扇区的视图（第 2b 步）：离线重放两种方案都不过，**不做** | `crowd3d/landmark_map.py`、`crowd3d/jobs.py`、BENCHMARK §6.7–6.9 |
| 结构优先接进命令行、基准、作业、API 和网页 | **采纳（管道）**，产出是素模 | 一个入口三处共用。但今天每个真实众包模型都过不了配准门，判定器不跑；而且只有一种可商用的天空掩码（A+R），按负责人问题 2 的默认答复保留"两种掩码"的规则，D25 不跑、D15 的天空轮廓线一路沉默。所以产品给出的是 OSM 分件的素模，标"素模（照片未对齐）" | `crowd3d/structure_pipeline.py`、`crowd3d/scene_io.py`、`JOB_STYLIZE`、`ModelInfo.stylized`、STRUCTURE_FIRST §2.9 |
| 留出集评测（commons-ten-v1 从没跑过的五栋）和评审的指标 | **采纳**，分层报告没做 | 划分文件、指标第 7 版、通过标准 `review-v1`、基线和冻结候选各跑一次留出集都做了。划分文件写的"没在 dev 出现过的摄影师"和"晚到的会话"两层，工具还不分开报 | `eval/splits/commons-ten-v1-holdout-v1.json`、`eval/criteria/review-v1.json`、`scripts/benchmark.py`、`scripts/eval_real_landmark.py` |
| 需要 GPU 的方法 | **现在不做** | 这台机器只有 4 核 CPU、没有 GPU，验证不了；先把 CPU 上能测的做完 | — |
| 学习型分割（天空、屋顶等） | **不进产品** | 许可：OneFormer 用的 ADE20K 只许非商用研究；Dm+R（U2-Net skyseg）训练数据不明；C+R 用的 SAM 训练集 SA-1B 只许研究用，要法务确认（STRUCTURE_FIRST §1、§2.3 的表）。`crowd3d` 只认 A+R，别的掩码名一律拒绝；Dm+R 只在仓库外的研究驱动里做对照 | `structure_pipeline.PRODUCT_MASKS` |
| 按表面的颜色门 | **现在不做** | 负责人的优先级：颜色不是重点，目标是结构框架（2026-10-24）。而且真实照片的模型一个都没过配准门，颜色门没有东西可放 | — |
| 生成式世界模型（生成的视图） | **不采纳** | 不变的规则：D8 接受"玩具"定位但配准门不放松，D26 只用量出来的值、不编造；生成的视图永远不算证据（STYLIZED_MODEL §2.3，ARCHITECTURE §3.4.3） | — |

**负责人问题**（按默认答复执行，负责人可以改）：

1. 主要指标是建筑局部坐标系（Sim3 对齐到参考）里 1.5 m 的支持覆盖 `cov_local_supported_m_1p5`；同时报告按份额的伴随数 `cov_local_majority_m_1p5`、地理 5 m（次要）和地理 1.5 m（只作参考）。
2. 只有一种可商用天空掩码：保留"两种掩码"的规则，产品为每个有分件快照的地标出 OSM 分件 GLB，配准被拒时整栋素模并标"素模（照片未对齐）"；只在没有重建作业排队时跑，1200 s 上限。
3. 续建的模型直接发布（同样的质量门）。续建没通过 dev 验证，所以一直关着，这条没有生效。

另外：任务 #87（照片位姿）只取了基准部分（第 6 版的 `photo_pose_*`、参考缓存、`rescore`），公开的位姿接口和网页等隐私决定。

## 2. 做了什么（按阶段的提交）

每一步都保持快速测试和 node 测试通过；基准从钉在候选提交的 worktree 跑。

| 阶段 | 提交 | 内容 |
|---|---|---|
| 0 共用的评测工具 | ef2815a、4b97916、d7e75fb、ef64dcd、84db9a1、fcfaeec、819cb78；修正 450b086、e207a0a、6980aeb | 照片位姿指标和 `rescore`；每栋建筑一份缓存的参考答案；`RECON_OUTPUT_VERSION` 和走产品路径的金样测试（替代按文件比对的补算守卫）；裁剪后的 OSM 分件快照（13 个地标）；开发 / 留出划分、分件快照进输入指纹、3 GiB 磁盘守卫；指标第 7 版（地理误差、两种坐标系的覆盖、标定、运行代价、版本）；`compare --criteria review-v1`、`--exclude-group`、分划分汇总、`posediff`；硬守卫看每一次重复；按份额的覆盖伴随数和每个模型画错的墙 |
| A 基线 B0 | 819cb78（外加一行没提交的测试流程补丁，见 3.2） | synthetic-v1、mdx-two-v1 新跑；v1-after-fixes 核心五栋补算到第 7 版；洗礼堂、亚琛新跑一次看运行时间；留出五栋第一次跑 SfM |
| B 地理锚点（只记录） | 9a433b3、25e67c7、ca700cc | `GeoAnchor` 存进模型 JSON（原有字段逐字节不变）；`georegister_inputs`、`rescore --regeo`、四个分量的 `calib_*`、`calibration` 子命令；在开发集上拟合、冻结下限 `dev-core5-v1` |
| C 结构库和基准 | c92d9fb、80b7a38、59266a7、5802b2f、8b4dd42 | `scene_io.py`、`stylize.register_scene`；`structure_pipeline.py`（只认 A+R）；`stylize_model.py` 默认走结构；基准经结构流水线风格化、填上 `struct_*`、`rescore --structure on\|off` |
| D 任务 #73 | a125c00、3457ae2 | 读锚点的吸附门（三个开关），`--snap-from` 复用旧吸附前先重查；dev 重放后开关默认打开 |
| 第一轮核查的修正 | 257fdf0、9473072、a1e8c8a | 退化相似变换（尺度 ≤ 0）不再让测试流程崩溃；`--snap-from` 只对流水线模型重查；锚点的倾斜 σ 不再随 SfM 坐标系变（`ANCHOR_METHOD` 2） |
| E 持久地图 | fa1807f、665a30c、33d6c1e、031773b、5b81284、be264f8、0520acd、76fb96a | 特征和匹配缓存（验证后默认开）；输入版本、唯一的替换规则、保存已发布模型的地图；`--max-requests`；续建（验证没过，默认关）；扇区视图的离线重放（2b 不做） |
| F 产品里的结构 | be2d00c、6727553 | 每个发布的模型排一个风格化作业，在子进程里跑结构流水线；地标页链接风格化版本并标出"素模（照片未对齐）" |
| G 地理信息和文档 | 4275e79、6416910 | API 的 `geo` 块、质量门里只记录不拦的锚点检查、"误差"改名"GPS 对齐残差"；ARCHITECTURE、BENCHMARK、STRUCTURE_FIRST、VISUAL_LOCALIZATION、STYLIZED_MODEL |
| 第二轮核查的修正 | 5e44646（= F） | 地标详情的作业列表只列重建作业；记下两处缺口（没有"重新风格化"入口；1a 的成本判据按 CPU 判） |
| H 最终评测（本文） | 本提交 | B0、F 和拆分的结果文件进 `eval/results/`，排行榜重新生成，本文 |

## 3. 评测方法

### 3.1 测试集和划分

- **dev**：commons-ten-v1 的核心五栋（洗礼堂、亚琛、勃兰登堡门、比萨斜塔、圣母教堂），加上整个 synthetic-v1 和 mdx-two-v1。所有调参都只在这里做。
- **test**：从没跑过 SfM 的五栋（圣家堂、匈牙利国会大厦、林肯纪念堂、印度门、悉尼歌剧院），513 张照片。只用事先写明的冻结提交跑：B0 一次、F 一次，每次两个重复。这一轮没有为留出结果改过代码或参数，划分仍是 `holdout-v1`。
- 详细规则、摄影师重叠和已知污染见 BENCHMARK §2。

### 3.2 基线 B0 和候选 F

| | B0 | F |
|---|---|---|
| 代码 | 819cb78：产品代码和评审之前（db06c3c）相同，加上第 0 阶段的评测工具；`scripts/eval_real_landmark.py` 有一行没提交的补丁（相似变换尺度 ≤ 0 时不算尺度误差，后来在 257fdf0 提交），产品代码没动 | 5e44646，工作区干净，全部默认值 |
| synthetic-v1、mdx-two-v1 | 新跑 `b0`（2 次重复） | 新跑 `f`（2 次） |
| 核心五栋 | v1-after-fixes（1d6055b 跑的 SfM）在 819cb78 上补算到第 7 版并重新风格化：`v1-after-fixes-s7`；另外洗礼堂、亚琛新跑 1 次：`b0-dev`（只用来看运行时间） | 新跑 `f-dev`（2 次），勃兰登堡门只传前 104 个请求、圣母教堂前 31 个（v1-after-fixes 两次都只做到这里），其余三栋全部请求 |
| 留出五栋 | 新跑 `b0-test`（2 次，每栋 3600 s 上限，五栋都传完了） | 新跑 `f-test`（2 次，同样的请求数 66/44/50/58/55，输入指纹相同） |
| 机器 | 4 核 CPU；留出运行期间约一半的分钟采样里有别的进程占 CPU（平均负载 5.0）（`$R/b0/logs/contention.log`） | 4 核 CPU；421 个分钟采样里没有别的进程超过 20% CPU，平均负载 1.97、最高 4.28（`$R/f/logs/contention.log`） |

两边都是同一时间只跑一个 COLMAP，代价用 CPU 秒（墙钟时间会被同时运行的进程拉长）。

### 3.3 指标和通过标准

- 指标定义见 BENCHMARK §7.8。主要指标按负责人问题 1：`cov_local_supported_m_1p5`。
- 通过标准 `review-v1`（BENCHMARK §6.2）：dev 上主要指标**显著变好**，test 上**不变差**；六个硬守卫在每栋、每次重复上都是 0；覆盖错误、朝向放反、照片位姿错误、严重错误相机在噪声内（汇总和每一栋）；偏航、尺度、水平位置、倾斜的 2σ 覆盖率在 [0.85, 0.99]、3σ 外不超过 5%、至少 20 个模型；`cpu_time_s` 等代价不超过 B0 的 1.25 倍，`jobs_cpu_s` 在 dev 上要显著变少。发布数只是背景。
- 显著性：差值超过 max(最小变化量, 相对下限 × 旧值, 两次重复的极差)。只有两次重复，噪声带会低估"两种结果都可能出现"的情形（BENCHMARK §8）。
- 留出集的标定另用设计里的规则：2σ 覆盖率在 [0.88, 0.99]，3σ 外不超过 5%（尺度 10%），**至少 3 栋留出建筑各有至少 3 个能核对的模型**，否则"证据不足"。

### 3.4 拆开看每个包：2×2×2

只有 B0、F 两种 SfM 配置真跑了，其余用 `rescore`（不重跑 SfM，同一份 5e44646 的代码打分）切换：

- **P1** = #73 的吸附门开关（`--snap-gate on|off`）。B0 的产物没有锚点，开的时候加 `--regeo` 补出锚点；关的时候用旧的 σ。
- **P3** = 结构（`--structure on|off`）。
- **P2** = 打分的是谁的 SfM 产物：B0 的还是 F 的。续建关着，所以 P2 就是特征缓存（加上 SfM 运行间的随机差别）。

8 个格子的结果在 `eval/results/<测试集>/ablation-v5/<b0|f>[-dev|-test]-r<P1><P3>.json`（不进排行榜）。

### 3.5 哪些是样本内的

- 锚点的下限 `dev-core5-v1` 是在核心五栋（v1-after-fixes 的产物）上拟合的，所以 dev 上的标定是**样本内**的：F 的核心五栋是新跑的 SfM，但建筑和照片相同。
- 吸附门的各个门槛、#73 的开关、缓存和续建的验证都在 dev 上定。
- 结构规则是在目录里 13 个地标的 OSM 快照上定的，包括留出五栋；照片位姿 EXIF 位置 σ 的标定照片也包括留出五栋（BENCHMARK §2 的污染清单）。
- 14 位摄影师两边都有，拍了留出集 513 张里的 53 张；工具还不按"没见过的摄影师"分开报（第 6 节），所以逐照片的数字只有全部照片的。
- 圣家堂的参考只从缓存读（原始参考删了）；悉尼歌剧院的参考过不了轮廓核验，没有参考答案；印度门一个模型都没发布，但它的参考有问题（41 台参考相机里 12 台在自己 EXIF GPS 的对称点上），它的 41 张照片仍按 EXIF 参与照片位姿（`photo_pose_*`）的合计；划分文件要求把印度门单独报告、不进合计，评测工具还没做到，所以 test 的照片位姿合计数含有这部分有问题的参考。

## 4. 结果

### 4.1 dev：B0 对 F

| 指标 | B0 | F | review-v1 |
|---|---|---|---|
| `cov_local_supported_m_1p5`（主要，m） | 134.85 | 136.8 | FAIL：+1.95，门槛 9.9，要求显著变好 |
| `cov_local_majority_m_1p5`（按份额的伴随数） | 114.9（补算） | 116.4 | 只报告 |
| `cov_geo_supported_m_5`（次要） | 65.9 | 70.9 | 只报告 |
| `cov_geo_supported_m_1p5`（参考） | 0 | 0 | 只报告 |
| 六个硬守卫 | 0（结构两项没有值） | 0 | PASS（每栋每次） |
| `cov_local_wrong_m_1p5` | 79.3 | 78.8 | PASS |
| `cov_geo_wrong_m_5` | 148.2 | 144.7 | **FAIL**：比萨 0.5 → 2.0（噪声 1.0） |
| `wrong_face_cameras_shown` / `_models_published` | 4 / 0.5 | 2 / 0 | PASS |
| `photo_pose_wrong` | 84 | 82.5 | PASS |
| `gt_gross_shown` | 7 | 6.5 | PASS |
| 标定 2σ 内（偏航 / 尺度 / 水平位置 / 倾斜） | 偏航 0.798、尺度 0.560（旧字段的 σ；没有另两项） | 50.5/53、50/53、50/53、52/53 | PASS【样本内】 |
| 标定 3σ 外 | 偏航 3.5、尺度 18 | 0、1、1、0 | PASS |
| `request_latency_p90_s` | 178.85 | 73.7 | PASS（0.41 倍） |
| `cpu_time_s`、`jobs_cpu_s`、峰值内存、`photo_time_to_pose_p50_s`、磁盘 | 没记 | 9,237 s、9,057 s、2,873 MB、81.7 s、776 MB | INSUFFICIENT（v1-after-fixes 没有这些数；成本另比，见 4.4） |
| `models_published`（背景） | 62.5 | 61 | 只报告 |
| `geo_camera_error_median_m`（参考） | 5.99 | 6.58 | 只报告 |

分建筑（B0 → F）：

| 建筑 | 主要指标 m | 地理 5 m 错误覆盖 m | 发布的模型 |
|---|---|---|---|
| 洗礼堂 | 35 → 35 | 22.5 → 16 | 6.5 → 6 |
| 亚琛 | 0 → 0 | 0 → 0 | 2 → 2 |
| 勃兰登堡门 | 12.45 → 13.9 | 27.9 → 28.4 | 22 → 22 |
| 比萨斜塔 | 9 → 8 | 0.5 → 2.0 | 12 → 11 |
| 圣母教堂 | 78.4 → 79.9 | 97.3 → 98.3 | 20 → 20 |

- 比萨那一条：核心五栋所有 SfM 运行里比萨的 `cov_geo_wrong_m_5` 是 v1 {1, 0}、1a 缓存 {0, 1}、不用缓存的对照 {2}、1b 续建 {2, 1}、F {2, 2}（`$R/f/NOTES.md`）。比萨的值在各次运行里有 0、1、2 三种（每次重复的 `cov_geo_wrong_m_5` / 发布数：v1 1/12、0/12；1a 缓存 0/11、1/12；不用缓存的对照 2/11；1b 续建 2/11、1/12；F 2/11、2/11），2.0 只出现在发布 11 个模型的运行里，但发布 11 个时也有 0 和 1。2.0 在不用缓存时也出现，两次重复的噪声带盖不住它。
- 同样来自建图随机性的显著变差（`$R/f/compare/commons-dev_rs_P2_b0r00_vs_fr00.txt`）：发布数 62.5 → 61、地理偏航误差中位数 3.87 → 4.71°（噪声 0.8）。缓存不改变建图的输入（BENCHMARK §6.7：46 个作业的数据库逐字节相同）。
- 换同一份代码打分（`ablation-v5/b0-dev-r00` 对 `ablation-v5/f-dev-r11`）得到同样的 FAIL 行。
- **synthetic-v1**：主要指标 102 = 102（FAIL，要求变好），吸附门 2 个通过、2 个都对、0 个错；`disk_run_mb` 8.3 → 18.4 MB（2.22 倍，保存的地图）FAIL；合成集没有分件快照，`struct_*` 没有值，标定只有 12 个模型，都是 INSUFFICIENT。**mdx-two-v1**：只有主要指标 FAIL（36 = 36），守卫、结构、代价都通过，磁盘 1.17 倍。

出处：`eval/results/commons-ten-v1/v1-after-fixes-s7.json`、`f-dev.json`、`ablation-v5/b0-dev-r00.json`；`eval/results/synthetic-v1/{b0,f}.json`、`eval/results/mdx-two-v1/{b0,f}.json`；比较输出 `$R/f/compare/{commons-dev,synthetic,mdx}_b0_vs_f.txt`、`commons-dev_rs_b0r00_vs_fr11.txt`。

### 4.2 test（留出五栋）：B0 对 F

| 指标 | B0 | F | review-v1 |
|---|---|---|---|
| `cov_local_supported_m_1p5`（主要） | 33 | 34 | PASS（不变差；+1，门槛 2） |
| `cov_local_majority_m_1p5` | 33（补算） | 34 | 只报告 |
| `cov_geo_supported_m_5` / `_1p5` | 0 / 0 | 0 / 0 | 只报告 |
| 硬守卫（风格化三项、`stale_overwrites`） | 0 | 0 | PASS（五栋） |
| `struct_incorrect`、`struct_qa_failed` | — | 0 | INSUFFICIENT：林肯纪念堂、印度门、悉尼歌剧院没有发布的模型，没有值 |
| `cov_local_wrong_m_1p5` / `cov_geo_wrong_m_5` | 4.5 / 37.5 | 3 / 37 | PASS |
| `wrong_face_*` | 0 / 0 | 0 / 0 | PASS |
| `photo_pose_wrong` | 37.5 | 38 | PASS（噪声 1） |
| `gt_gross_shown` | 1.5 | 2.5 | PASS（门槛 3） |
| 标定 | 4 个模型，只有偏航和尺度（旧的 σ 没有水平位置和倾斜） | 4 个模型，四个分量 | INSUFFICIENT（少于 20） |
| 尺度 2σ 内 / 3σ 外 | 0.125 / 2.5（旧字段的 σ） | 1.0 / 0（锚点） | 同上 |
| `models_published`（背景） | 5.5 | 6 | 只报告 |
| `geo_camera_error_median_m` | 15.78 | 15.07 | 只报告 |

分建筑：圣家堂主要指标 33 → 34 m、发布 1 → 1 个、地理相机误差 3.58 → 3.60 m；匈牙利国会大厦主要指标 0 → 0、发布 4.5 → 5、地理相机误差 27.98 → 26.55 m；林肯纪念堂、印度门、悉尼歌剧院两边都没发布模型。地理偏航误差中位数 1.375 → 2.67°（只有 4 个模型，`compare` 判为显著变差）。

出处：`eval/results/commons-ten-v1/b0-test.json`、`f-test.json`、`ablation-v5/b0-test-r00.json`；`$R/f/compare/commons-test_b0_vs_f.txt`、`commons-test_rs_b0r00_vs_fr11.txt`（同一份代码打分，同样只有 INSUFFICIENT 行）。

### 4.3 拆开看：P1 吸附门、P3 结构、P2 缓存

**P1 和 P3 没有改变任何一个配准结果**：

| 测试集 | 8 个格子里吸附门的通过 | P1 改了拒绝理由的模型 | P3 改了的配准结果 |
|---|---|---|---|
| synthetic-v1 | 每格 2 个，都对，0 个错 | 24 个里 1 个；`style_snap_refused_gps` 1 → 1.5 | 0 |
| mdx-two-v1 | 每格 0 | 8 个里 0 个 | 0 |
| commons 核心五栋（F 的产物） | 每格 0 | 122 个里 7 个；偏航窗口 ±16.2° → ±25.8°（锚点的 σ）这类只改数字的不算 | 0 |
| commons 留出五栋（F 的产物） | 每格 0 | 12 个里 1 个 | 0 |

- P1 的好处（对抗用例里错误通过 69 → 0，REAL_WORLD_EVAL §7 第 1 条）在这里看不出来：没有一个众包模型走到门的"通过"分支。P1 唯一显著的代价是 dev 的风格化时间 92.85 → 121.2 s（`$R/f/compare/commons-dev_rs_P1_fr00_vs_fr10.txt`）。
- B0 的产物在 P1 开时用补出的锚点，dev 的偏航 2σ 覆盖率因此 0.798 → 0.963：这是锚点和下限的作用，不是吸附门的。

**P3（结构）的代价**，同一份 F 的产物，结构关 → 开：

| 范围 | 体块的墙面数 | GLB（kB，gzip） | 风格化时间 s | 结构这一步 s | 结构峰值内存 MB |
|---|---|---|---|---|---|
| 核心五栋 | 79 → 2,375 | 11.5 → 204.4 | 92.85 → 276.2 | 103.2 | 364.9 |
| 留出五栋 | 97 → 1,521 | 19.45 → 127.85 | 11.6 → 70.6 | 19.1 | 403.65 |
| mdx-two-v1 | 40 → 192 | 10.8 → 38.3 | 3.95 → 7.2 | 4.8 | 260.9 |

结构的守卫在每格都是 0：`struct_incorrect`、`struct_unbacked`、`struct_adopted_unregistered`、`struct_qa_failed`、`style_qa_failed`；`struct_checked`、`struct_d15_adopted`、`struct_photo_share` 也都是 0（什么都没判）。每个模型都在 1200 s / 4 GB 之内。

**P2（缓存，续建关）**，F 的产物对 B0 的产物、P1 和 P3 都关：

| | 核心五栋 B0 → F | 留出五栋 B0 → F |
|---|---|---|
| 主要指标 m | 134.85 → 136.8 | 33 → 34 |
| 按份额的伴随数 m | 114.9 → 116.4 | 33 → 34 |
| 地理 5 m 支持覆盖 m | 65.9 → 70.9 | 0 → 0 |
| 建筑局部 1.5 m 错误覆盖 m | 79.3 → 78.8 | 4.5 → 3 |
| 地理 5 m 错误覆盖 m | 148.2 → 144.7 | 37.5 → 37 |
| 发布的模型 | 62.5 → 61 | 5.5 → 6 |
| 作业 CPU 秒 | 没记 → 9,057 | 14,859 → 4,127 |

**交互**：P3 在 B0 和 F 的产物上作用相同。P1 不完全相同：在 B0 的产物上打开 P1 会加 `--regeo`，所以同时改变标定（dev `calib_yaw_cover2` 0.798 → 0.963，F 一直是 0.953）；P1 改变的拒绝理由类别在 B0 的 test 模型上是 0/11，在 F 上是 1/12。两者都不改变任何配准结果。

出处：`eval/results/*/ablation-v5/*.json`（32 个格子），汇总表 `$R/f/compare/table2x2.md`，拒绝理由 `$R/f/compare/reasons_<测试集>_<前缀>.json`，各对比较 `$R/f/compare/*_rs_P{1,2,3}_*.txt`。

### 4.4 代价

| 范围 | 对照 | 对照的作业 CPU 秒 | F | 变化 | 作业墙钟时间 |
|---|---|---|---|---|---|
| synthetic-v1 | B0 新跑（2 次） | 165.05 | 112.4 | −32% | 66.05 → 48.95 s |
| mdx-two-v1 | B0 新跑（2 次） | 156.55 | 111.5 | −29% | 66.1 → 50.0 s |
| 核心五栋 | 同一请求前缀、不用缓存的对照（031773b，各 1 次） | 22,497 | 9,057 | **−59.7%** | 7,475 → 4,547 s（−39%） |
| 洗礼堂 + 亚琛 | B0 新跑（`b0-dev`，1 次） | 1,825.5 | 745.45 | −59% | 691.9 → 337.9 s |
| 留出五栋 | B0（2 次，当时平均负载约 5） | 14,859 | 4,127 | −72% | 5,360 → 1,815 s（−66%） |

- 核心五栋分建筑（对照 → F）：洗礼堂 427 → 306 s，亚琛 1,071 → 440 s，勃兰登堡门 8,996 → 2,033 s，比萨 5,851 → 2,442 s，圣母教堂 6,154 → 3,835 s。和 v1-after-fixes 比作业墙钟时间是 9,409.5 → 4,547.1 s（−52%），但 v1 当时的机器负载不同，所以成本只和同一提交的对照比。
- 留出五栋的 −72% 里有一部分是 B0 当时机器忙；干净的数是核心五栋的 −59.7%。圣家堂每次运行的墙钟时间 3,002.65 → 962.65 s（中位数）。
- 留出五栋的其他代价（F / B0）：`cpu_time_s` 0.284 倍，峰值作业内存 3,162 → 2,734 MB，请求延迟 90 分位 74.05 → 20.8 s，照片拿到位姿的中位时间 714 → 241 s，磁盘 531 → 550 MB（1.03 倍）。全部通过 1.25 倍的上限。
- 磁盘变大的地方：保存的地图。synthetic-v1 8.3 → 18.4 MB（这一条按 1.25 倍判是 FAIL），mdx-two-v1 33.3 → 39.1 MB。
- 产品里的结构风格化作业（第 F 阶段，v1-after-fixes 核心五栋第 1 次的 20 个最新模型）：20 个都出了 GLB、都是素模，子进程 CPU 中位数 4.9 s、最多 19.7 s，峰值内存不超过 300 MB，所以一个风格化作业最多让新排进来的重建等约 20 s（`$R/F-product-structure/measure/measure.json`）。

出处：`eval/results/{synthetic-v1,mdx-two-v1}/{b0,f}.json`、`eval/results/commons-ten-v1/{b0-dev,b0-test,f-dev,f-test}.json`；不用缓存的对照 `$R/E1-E5-cache/results/commons-ten-v1/e5-nocache-ctl.json`（洗礼堂、亚琛、勃兰登堡门）和 `e5-nocache-ctl2.json`（比萨、圣母教堂）；`$R/f/compare/commons-dev_e5nocachectl_vs_f.txt`、`commons-dev_b0freshBaptAachen_vs_f.txt`、`commons-test_rs_P2_b0r00_vs_fr00.txt`。

### 4.5 标定

**dev【样本内】**：上面 4.1 的 review-v1 行全部通过。按 `calibration` 去重以后（同一个模型两次重复只算一次）是 82 个模型，合在一起的 2σ 覆盖率：偏航 0.939、尺度 0.951、水平位置 0.951、倾斜 0.976（`$R/f/compare/calib_f-dev.txt`）。

**test（留出）**：在冻结的下限 `dev-core5-v1` 上算，没有在留出集上重新拟合。去重后 8 个模型：圣家堂 2 个、匈牙利国会大厦 6 个。

| 分量 | 1 / 2 / 3σ 内（合在一起） | 3σ 外 | σ 中位数 | 按建筑加权 |
|---|---|---|---|---|
| 偏航 | 0.875 / 1.0 / 1.0 | 0 | 7.62° | 0.916 / 1.0 / 1.0 |
| log 尺度 | 1.0 / 1.0 / 1.0 | 0 | 0.356 | 1.0 / 1.0 / 1.0 |
| 水平位置（枢轴的东、北） | 0.25 / 0.5 / 0.5 | 0.5 | 4.11 m | 0.5 / 0.666 / 0.666，3σ 外 0.334 |
| 倾斜 | 1.0 / 1.0 / 1.0 | 0 | 2.36° | 1.0 / 1.0 / 1.0 |

- 误差中位数：偏航圣家堂 1.3°、匈牙利国会大厦 5.0°；log 尺度 0.11、0.27。
- **匈牙利国会大厦的水平位置说小了**：6 个模型里 4 个在 3σ 外，误差中位数 17.3 m，σ 只有 4.1 m；圣家堂的 2 个都在 1σ 内。
- 偏航 2σ 覆盖率的 95% Wilson 区间 0.676–1.0。
- 把 B0 的留出产物按现在的下限补出锚点（`ablation-v5/b0-test-r11`，`--regeo`）一起算：14 个模型（圣家堂 2、匈牙利国会大厦 12），偏航、尺度、倾斜 2σ 内都是 1.0，水平位置 0.143 / 0.357 / 0.357、3σ 外 0.643；仍然只有一栋建筑有 3 个以上的模型。
- **结论：四个分量都是"证据不足"**（设计的规则要至少 3 栋留出建筑各有至少 3 个模型，只有匈牙利国会大厦一栋）。`review-v1` 的标定行也是 INSUFFICIENT（4 < 20）。
- **后果**：把 `dev-core5-v1` 加进 `Settings.calibrated_anchor_ids`、在地标页显示"位置 ±X 米（1σ）"的前提是留出标定通过，它没通过；而唯一有数据的留出建筑和水平位置的 σ 相矛盾，所以位置文字继续关着。顺便一提，即使打开，dev 上它会显示的水平 σ 中位数是 16.9 m（7.7–47.1 m），大部分来自尺度项（`$R/G-geo-docs/geo_text_v1.json`，v1-after-fixes 核心五栋最后显示的 38 个模型）。

出处：`eval/results/commons-ten-v1/f-test.json`、`ablation-v5/b0-test-r11.json`；`$R/f/compare/calib_f-test.txt`、`calib_verdict_f-test.json`、`calib_verdict_f-test+b0-test-regeo.json`（判定工具 `$R/f/tools/calib_verdict.py`，同时给出按设计规则和按 `review-v1` 的 Wilson 读法的结论，两种都是证据不足）。

### 4.6 不设上限的吞吐（持久地图验证计划里的一项）

F、默认配置、整个到达计划不截断、上限 14,400 s、风格化开，勃兰登堡门和圣母教堂各 1 次，都正常结束。"第一个发布模型的时间"按到达计划的处理时间算：到发布它的那个请求为止，各请求接收和作业秒数的累计。

| | 勃兰登堡门 | 圣母教堂 |
|---|---|---|
| 请求（照片） | 105（150） | 43（111） |
| 截断 | 否 | 否 |
| 墙钟时间 | 1,345 s | 3,953 s |
| 作业 CPU 秒 | 2,253 | 7,641 |
| 每小时处理的照片 | 410 | 106 |
| 请求延迟 50 / 90 分位 | 13.1 / 73.2 s | 137 / 382 s |
| 照片拿到位姿的中位时间 | 69 s | 104 s |
| 峰值作业内存 | 2,838 MB | 2,977 MB |
| 第一个发布的模型 | 东立面，第 43 个请求，98.9 s 后 | 东北立面，第 4 个请求，6.0 s 后 |
| 发布 | 22 个模型，5 个立面 | 26 个模型，5 个立面 |
| 完整模型 | 没发布 | 没发布 |

- 主要指标：勃兰登堡门 11.9 m、圣母教堂 77.9 m；硬守卫都是 0。
- 验证计划的"第一个发布模型 ≤ 2 小时"两栋都远远满足。作为对照，v1-after-fixes 两次都在 3600 s 的上限里被截断（勃兰登堡门做到第 104 个请求、圣母教堂第 31 个，BENCHMARK §6.1）。

出处：`eval/results/commons-ten-v1/f-uncapped.json`；累计时间来自运行目录的 `summary.json`（`$R/f/compare/throughput.json`，工具 `$R/f/tools/throughput.py`）。

## 5. 没变好的地方，为什么

1. **模型在地图上的绝对位置没变。** 地理相机误差中位数 dev 5.99 → 6.58 m、test 15.78 → 15.07 m（匈牙利国会大厦 27.98 → 26.55 m），地理 1.5 m 的支持覆盖在真实照片上全是 0。采纳的四个包都不带来新的偏航、尺度或位置来源：锚点只是把已有的地理配准记下来并给出 σ；缓存不改变建图的输入；结构在配准之后才起作用。所以配准门也一样：真实照片的模型 dev 58 个可参评、test 6 个，一个都没通过。能改这件事的只有独立的偏航和尺度来源，比如视觉定位（第 6 节）。设计阶段就预见到这一点，所以负责人问题 1 把主要指标放在建筑局部坐标系里。
2. **主要指标也没有显著变好。** 缓存让建图读到逐字节相同的数据库，局部形状只能随 SfM 运行间的随机性变；+1.95 m 在 9.9 m 的噪声内。dev 要求的"显著变好"这些包本来就给不了。
3. **产品里的结构只出素模，照片量出的数字是 0。** 两个原因：
   - 每个真实众包模型都过不了配准门，判定器不跑（`struct_checked`、`struct_d15_adopted`、`struct_photo_share` 在 dev 和 test 的每个格子里都是 0）；
   - 只有一种可商用的天空掩码。参考场景（照片放得准）只用 A+R 时，洗礼堂和亚琛的照片数字也都是 0：D15 采纳 0 处、核验 0 处，D25 不跑；同样的场景加上研究用的 Dm+R，洗礼堂的外轮廓按 k 0.8178 采纳、D24 的数字里 42% 来自照片，亚琛唱诗堂按 D25 至少 36.255 m（`$R/v0/out/`）。照片数字全部依赖第二种掩码。
   - 所以今天产品得到的是：正确的 OSM 分件体块、管道、标签，以及更大的 GLB 和更长的风格化时间（4.3 节）。
4. **续建一直关着。** dev 验证没过：版本连续性 56%（要求 90%），和从零重建的位姿差五栋里两栋超出，先试续建再整个从零重建让净省的 CPU 为零（BENCHMARK §6.8）。负责人问题 3 因此没有生效。
5. **"一张图切出各扇区视图"（2b）不做**：离线重放里单用视图发布的立面少，混合方案多显示看错立面的相机（BENCHMARK §6.9）。
6. **留出集的标定证据不足**（4.5 节），位置文字不能打开。
7. **几个次要指标因为建图随机性显著变差**：dev 发布数 62.5 → 61、地理偏航 3.87 → 4.71°；test 地理偏航 1.375 → 2.67°（4 个模型）。dev 的两项在不用缓存的同提交对照里也出现过（对照发布 61 个，偏航见 BENCHMARK §6.7），不是缓存造成的。test 没有不用缓存的同提交对照：匈牙利国会大厦的偏航在 B0 是 1.28° / 1.50°，F 是 2.81° / 5.21°，这些值在不用缓存的运行里没出现过，只有 4 个模型，说不清原因。

## 6. 还没解决的问题和下一步

1. **视觉定位**（[VISUAL_LOCALIZATION.md](VISUAL_LOCALIZATION.md)，只是设计）：新照片对已经配准好的地标级模型做绝对定位，给出不靠 OSM 轮廓的偏航和尺度。它是唯一能改变绝对位置和配准门结果的路。这一轮做好了它要的底子：每个已发布模型的稀疏地图存在 `data/maps/<地标>/models/` 下，锚点是独立的记录（VISUAL_LOCALIZATION §5.1 列的要持久保存的东西）。
2. **单掩码的决定（D28）**：要么保留"两种掩码"（现在的默认），要么让 A+R 单独算一路、门槛更严、G5 只在 A+R 上量，后者要有自己的一轮验证（STRUCTURE_FIRST §2.9）。现在的代码在"只有一种天空掩码"那里就停了，不报 A+R 自己的读数，所以先要一个研究开关把这些读数报出来才能权衡。另一条路是查清 Dm+R 或 C+R 的许可，得到第二种可商用的掩码。
3. **需要 GPU 的阶段**：这台机器上验证不了。到有 GPU 的机器上时，同一个划分和 `review-v1` 可以直接拿来比；留出五栋已经按 B0 和 F 各跑过一次，再跑一个冻结的候选要按划分规则记录。
4. **留出证据太少**：五栋里三栋一个模型都不发布，标定要求的"至少 3 栋、每栋至少 3 个模型"凑不出来。要得出标定的结论，得加建筑（发一个新的测试集或划分版本）。
5. **评测工具的缺口**：
   - 逐照片指标还不按"没见过的摄影师"（留出集 458 张）和"晚到的会话"分开报；
   - 同一个测试集目录里同时有 dev 和 test 的结果时，`compare` 把所有同版本结果的汇总噪声都算进门槛，另一个划分的汇总极差也会混进来（结论这次没变，门槛变了）；要复现这里的数，比较时加 `--results-dir <空目录>`，只用两个结果自己的噪声块（第 7 节）；
   - 磁盘守卫只在每栋建筑开始前查：剩余空间两次低于 3 GiB 的规则，一共 9 个一分钟采样：10:29–10:33（F 的圣家堂第 1 次运行，最低 2.83 GiB）和 12:20 前后（不设上限的圣母教堂运行，最低 2.82 GiB）。
6. **产品的缺口**：没有"重新风格化"的入口（这个功能上线以前发布的模型、分件快照或风格化代码改了以后的模型，要等下一次发布才有新的 GLB，ARCHITECTURE §11.1）；完整模型一次都没发布过却占大量作业时间（REAL_WORLD_EVAL §7 第 4 条）；续建重试前可以考虑的改动见 BENCHMARK §6.8 末尾。
7. **照片位姿的公开接口**（任务 #87）等隐私决定。

## 7. 怎么复现

```bash
cd /home/user/temp && . .venv/bin/activate
mkdir -p /tmp/empty-results     # 比较时不读 eval/results 里其他运行的噪声块，和 V5 当时一样

# dev 和 test 的通过标准
python -I scripts/benchmark.py compare eval/results/commons-ten-v1/v1-after-fixes-s7.json \
    eval/results/commons-ten-v1/f-dev.json --criteria review-v1 --results-dir /tmp/empty-results
python -I scripts/benchmark.py compare eval/results/commons-ten-v1/b0-test.json \
    eval/results/commons-ten-v1/f-test.json --criteria review-v1 --results-dir /tmp/empty-results
python -I scripts/benchmark.py compare eval/results/synthetic-v1/b0.json eval/results/synthetic-v1/f.json \
    --criteria review-v1 --results-dir /tmp/empty-results

# 拆分：例如核心五栋上 P1（吸附门）的作用
python -I scripts/benchmark.py compare eval/results/commons-ten-v1/ablation-v5/f-dev-r00.json \
    eval/results/commons-ten-v1/ablation-v5/f-dev-r10.json --results-dir /tmp/empty-results

# 标定表
python -I scripts/benchmark.py calibration eval/results/commons-ten-v1/f-test.json --split test
python -I scripts/benchmark.py calibration eval/results/commons-ten-v1/f-dev.json --split dev
```

在开发机上，这几条的输出和 `$R/f/compare/` 里的逐字节相同。不加 `--results-dir` 时结论一样，门槛不同（6 节第 5 条）。

重跑本身（要先物化照片和参考，见 BENCHMARK §3；从钉在 5e44646 的 worktree 跑，同一时间只跑一个 COLMAP）：

```bash
python -I scripts/benchmark.py run --testset synthetic-v1 --config default --name f --repeat 2
python -I scripts/benchmark.py run --testset mdx-two-v1 --config default --name f --repeat 2
python -I scripts/benchmark.py run --testset commons-ten-v1 --config default --name f-dev --repeat 2 --time-cap 3600 \
    --max-requests brandenburg-gate=104 frauenkirche-dresden=31 \
    --buildings florence-baptistery aachen-cathedral brandenburg-gate leaning-tower-pisa frauenkirche-dresden
python -I scripts/benchmark.py run --testset commons-ten-v1 --config default --name f-test --repeat 2 --time-cap 3600 \
    --buildings sagrada-familia hungarian-parliament lincoln-memorial india-gate sydney-opera-house
python -I scripts/benchmark.py run --testset commons-ten-v1 --config default --name f-uncapped --time-cap 14400 \
    --buildings brandenburg-gate frauenkirche-dresden
# 拆分的一个格子（要源运行的产物还在 --work-root 下）
python -I scripts/benchmark.py rescore --testset commons-ten-v1 --from eval/results/commons-ten-v1/f-dev.json \
    --name f-dev-r00 --restylize --structure off --snap-gate off
```

- 留出五栋按规则只跑这两次（B0、F）；再跑要记成新的冻结候选。
- 每个结果文件的 `provenance.command` 记着当时的完整命令（含 `--work-root`、`--reference-cache`、`--note`），`provenance.rescored_from` 记着补算的来源结果和它的 sha256；仓库里的副本和当时写出的文件逐字节相同。

## 附录：结果文件

| 文件 | 是什么 |
|---|---|
| `eval/results/synthetic-v1/b0.json`、`eval/results/mdx-two-v1/b0.json` | B0 新跑（819cb78，2 次） |
| `eval/results/synthetic-v1/v1-merged-s7.json`、`eval/results/mdx-two-v1/v1-merged-s7.json` | v1-merged 在 819cb78 上补算到第 7 版并重新风格化（B0 的一部分） |
| `eval/results/commons-ten-v1/v1-after-fixes-s7.json` | 核心五栋的 B0：v1-after-fixes 的产物在 819cb78 上补算到第 7 版并重新风格化 |
| `eval/results/commons-ten-v1/b0-dev.json` | B0 新跑洗礼堂、亚琛各 1 次（看运行时间） |
| `eval/results/commons-ten-v1/b0-test.json` | B0 新跑留出五栋（2 次） |
| `eval/results/synthetic-v1/f.json`、`eval/results/mdx-two-v1/f.json`、`eval/results/commons-ten-v1/f-dev.json`、`f-test.json` | F 新跑（5e44646，2 次） |
| `eval/results/commons-ten-v1/f-uncapped.json` | F，勃兰登堡门和圣母教堂不设请求上限各 1 次 |
| `eval/results/<测试集>/ablation-v5/*.json` | 2×2×2 的 32 个补算格子（5e44646 打分），名字是 `<b0\|f>[-dev\|-test]-r<P1><P3>` |

派生结果（仓库外，`$R/` = `/tmp/claude-0/-home-user-temp/04ee4750-a5e3-5a4d-b130-9dfde89f5621/scratchpad/review/`）：`$R/f/compare/`（比较输出、标定表和判定、2×2×2 表、拒绝理由、吞吐量）、`$R/f/NOTES.md`（运行记录）、`$R/f/v5_summary.json`（汇总）、`$R/b0/NOTES.md`、`$R/E1-E5-cache/`、`$R/E6-E8-continue/`、`$R/F-product-structure/`、`$R/G-geo-docs/`、`$R/v0/`。
