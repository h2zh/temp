# 基准结果（benchmark results）

`eval/results/<testset_id>/<run_name>.json` 是 `scripts/benchmark.py run` 写出的结果，**只含元数据**（数字、路径、哈希），
照片、点云、截图等由照片衍生的东西留在仓库外的 `--work-root`（结果里只记路径）。`LEADERBOARD.md` 由
`python -I scripts/benchmark.py report` 重新生成，不要手工改。

## 怎么用

```bash
. .venv/bin/activate
python -I scripts/benchmark.py prepare --testset synthetic-v1                      # 合成集：本地生成，不下载
python -I scripts/benchmark.py prepare --testset mdx-two-v1 \
    --mdx-local florence-baptistery=<MegaDepth-X 解压出的 Baptistry_(Florence)/1> aachen-cathedral=<.../Aachen_Cathedral/1>
python scripts/materialize_testset.py commons-ten-v1 --cache-dir <data-root> --gt  # Commons 集（另一个脚本）

python -I scripts/benchmark.py run --testset synthetic-v1 --config default --name default-1
python -I scripts/benchmark.py run --testset mdx-two-v1 --config default --name default-noise --repeat 3
python -I scripts/benchmark.py compare eval/results/mdx-two-v1/default-noise.json eval/results/mdx-two-v1/<新运行>.json
python -I scripts/benchmark.py report
```

- `--data-root`（默认 `$CROWD3D_TESTSET_ROOT`）：物化后的测试集，`<data-root>/<testset_id>/<slug>/<photos[].file>`；
  参考模型在 `<data-root>/<testset_id>/<slug>/gt/<component>/sparse` 或 `<data-root>/mdx/<slug>/<component>/sparse`。
- `--work-root`（默认 `$CROWD3D_BENCH_DIR`）：每栋建筑一个子目录，含 Crowd3D 数据目录、`summary.json`、`summary.md`、
  `topdown.png`、`harness.log`。
- 每栋建筑在独立进程里跑，`--time-cap`（默认 3600 s）是硬上限；到 80% 时不再开始新的上传请求。
- 新的重建器：在 `crowd3d.reconstruct.register_reconstructor` 注册，再写一个 `eval/configs/<id>.json`
  把 `"reconstructor"` 设成它的名字，就能在同样的测试集上跑、和旧结果比较。

## 配置（eval/configs）

`{"config_id", "description", "reconstructor", "settings": {Settings 字段: 值}, "params": {crowd3d.params.Params 字段: 值},
"harness": {"arrival": "session"|"photo"|"single", "seed": 0}}`。名字不在白名单里（`params.SETTINGS_NAMES`、
`params.PARAM_NAMES`）就报错。`default.json` = 当前产品默认值（基线）；`fast.json` 只把处理图长边从 1600 px 降到 1024 px。

## 结果文件的字段

| 字段 | 内容 |
|---|---|
| `result_schema_version`, `metric_schema_version` | 结果格式和指标定义的版本；`compare` 遇到不同的指标版本会拒绝（`--force` 才比） |
| `testset` | `id`、`sha256`（testset.json 文件字节的 sha256）、各建筑的完整性（ok / changed / missing 的数量和名单，轮廓、参考模型的状态）、`on_changed`（变了的照片是跳过还是照用）、`materialize_testset.py` 的报告摘要 |
| `config` | `id`、配置文件 sha256、覆盖项，以及**完整的生效参数**（重建器、所有可调 Settings 字段、所有 Params 字段、harness 选项）和它的 sha256 |
| `provenance` | git 提交和 dirty 标志（改动的文件、`git diff HEAD` 的 sha256）、Python/pycolmap/OpenCV/numpy/Pillow 版本、CPU 数、开始时间、总用时、时间上限、重复次数、命令行 |
| `buildings.<slug>` | `status`（ok / failed / timeout / skipped）、`metrics`（全部指标；`--repeat` 时是各次运行的中位数，第 1 次的原值在 `metrics_first`，`details`/`artefacts` 也属于第 1 次）、`inputs_sha256`（实际用到的照片字节、轮廓、参考模型的指纹）、`details`（漏斗、扇区、作业、每个模型的质量门原因、轮廓拟合、GT 误差）、`artefacts`（仓库外的路径）；`--repeat` 时还有每次的 `repeats` |
| `aggregate` | 各指标按定义的方式（sum / mean / median）跨建筑汇总 |
| `noise` | 仅 `--repeat ≥ 2`：每栋建筑每个指标的极差（max − min）、跨建筑的最大值、汇总值的极差、哪些指标在重复运行中时有时无 |

## compare 怎么判断"真的变了"

- 两个结果的测试集 sha256 或指标版本不同：拒绝（`--force` 才比，并给出警告）。
- 某栋建筑两次的 `inputs_sha256` 不同（有照片变了或缺了）：单独列出，不进汇总比较。
- 每个指标：差值超过 max(该指标的最小变化量, 相对下限 × 原值, `--noise-factor` × 记录的运行间极差) 才算显著；
  极差取自两个结果本身、`--noise` 指定的文件和同一测试集下所有带 `noise` 的结果中的最大值。
- 方向来自指标定义：↑ 越大越好、↓ 越小越好、"-" 只是背景信息（只报"变了"）。
- 重复次数少时，极差会低估噪声（见下面的实测例子），所以噪声基线用 5 次重复。

## 风格化指标（metric_schema_version = 2）

`benchmark.py run` 默认对每个**已发布**的模型（含后来被替换的）跑一次风格化（`crowd3d.stylize`，和
`scripts/stylize_model.py --job-dir` 相同：读保留下来的作业目录 `sparse/0` + `images/` 和模型 JSON 里的相机 ENU），
为此 harness 打开 `Settings.keep_work_dirs`（只删 COLMAP 数据库和其他建图尝试）。`--no-stylize` 关掉它，style_* 指标为空。
GLB 和风格化摘要写在 `--work-root/<testset>/<run>/<slug>/style/<NN>-<kind>-<sector>/`，不进仓库。

- **配准门**（docs/STYLIZED_MODEL.md §3.2）：吸附到 OSM 轮廓后墙点中位距离 ≤ 1.5 m，并且逐边核验通过（点数达到中位边
  25% 的边才投票，≥ 60% 的投票边中位距离 < 3 m，通过的边方向至少差 30°）。只有 `georegistered = true`
  （`georeg_level = full`）的模型算"可参评"（eligible）；`prior_only` 的模型照样风格化并记录，但不计入通过率。
- `style_registration_pass_rate` = 通过 / 可参评，跨建筑**按模型合并**（`aggregate = "pooled"`：分子分母各自求和再相除），
  不是各建筑比率的平均。
- **通过的对不对**：有参考位姿（MegaDepth-X 或合成集真值）时，参考模型先用测试照片自己的 EXIF GPS/罗盘配准到 ENU，
  再把参考墙点拟合到 OSM 轮廓（刚体 + 有界尺度），得到"轮廓校正参考"（参考自身也要过同一个逐边核验，过不了的分量不用）。
  通过的模型在吸附之后还需要一个相似变换（吸附后的相机中心 → 参考相机中心）。metric_schema_version 4 起：它转角 ≤ 5°、
  把模型墙点（吸附后落在轮廓 3 m 内的点）挪动的 90 分位 ≤ 1.5 m、尺度差 ≤ 10%，才算对，否则记为"错误通过"
  （`style_registration_wrong`）。判定在墙上量：相机离墙 40–250 m，绕墙转 1° 相机就挪 0.7–4 m，照片在墙上却几乎不动；
  相机中心误差另记（`style_ref_camera_error_m` 中位、`style_ref_camera_error_worst_m` 最大），不进判定。
  schema 3 及以前的规则是"旋转 ≤ 5°、墙点中位处 ≤ 3 m、尺度差 ≤ 15%"，太松（圣母教堂 SW#20 还要 1.14 倍尺度、相机误差
  6.7 m，在旧规则下算对）。少于 3 台相机有参考位姿时不下结论。参考自己的误差：近圆轮廓（比萨斜塔）上参考模型自己的旋转
  也定不下来；合成集用的是生成器真值，没有这个问题。
- **不可参评却过了门**（`style_gate_passed_not_eligible`，schema 4）：`prior_only` 的模型（没有 GPS 配准）照样风格化，
  但必须整栋素模；这个计数必须为 0。schema 3 的运行里圣母教堂 SW#17 就是这样上了 4 面墙。
- **失败原因**（每个未通过的可参评模型，`details.models[].style.diagnosis.cause`）：`single_wall`（只有一条边有点，或通过的边
  全平行）、`too_few_points`、`symmetric_alias`（不加约束的 ICP 收敛到离正确修正 > 10° 的解）、`georegistration`（正确的修正
  本身就超出吸附上限 20° / 20 m）、`osm_outline`（参考模型自己都贴不上轮廓）、`residual`（方向对了但墙仍离轮廓太远）。
  证据优先用参考模型；没有参考时用 harness 自己的轮廓拟合（`outline_check`，对称建筑上只是下限）。
  metric_schema_version 3（v1-after-fixes 起）的吸附有了自己的拒绝理由，原因相应增加：`ambiguous`（另一个位置拟合得差不多）、
  `gps_disagrees`（吸附后相机到墙的距离和照片自己的 GPS 不符）、`shape`（轮廓要的尺度修正顶到 ±15%）、`scale_uncertain`
  （单面模型的 GPS 尺度不确定度 > 15%）；`georegistration` 改为"正确的修正在吸附的偏航窗口或 25 m 平移范围之外，或最优解落在
  窗口边上"；`symmetric_alias` 只剩诊断意义（不加约束的 ICP 会掉进的别名解，窗口把它挡在外面）。schema 4 再加
  `yaw_unverifiable`（近圆轮廓只搜平移，而配准报的 3σ_θ > 5° 或会让照片绕轮廓挪 > 1.5 m）和 `loose_fit`（吸附后墙点到轮廓的
  平均距离 > 0.45 m，轮廓定不住偏航）；`ambiguous` 也包括"偏航谷太平"（绕墙转 5° 仍拟合得一样好）。
- 每栋建筑的"最佳模型"（用户最多能看到的）：最新发布的模型里上色墙最多的那个，其次看观测面积；`style_walls_*`、
  `style_observed_area`、`style_fill_m2`、`style_glb_kb_gz` 都描述它。墙的状态按 §3.3：已解锁 = 掩码 ≥ 60% 且 ≥ 3 台相机
  各看到 ≥ 15%；其余上了色的是部分解锁；其他是素模（待解锁）。
- `style_qa_failed`、`style_provenance_violations` 必须为 0（诚实 QA、`audit_model`、`audit_glb`）。

## commons-ten-v1：本轮只记录了核心五栋

`commons-ten-v1/v0-baseline.json` 只跑了五栋：florence-baptistery、aachen-cathedral、brandenburg-gate、
leaning-tower-pisa、frauenkirche-dresden（`--buildings`）。其余五栋（sagrada-familia、hungarian-parliament、
lincoln-memorial、india-gate、sydney-opera-house）这一轮**没有跑**：容器大约每 3 小时重启一次，一栋大楼要 45–50 分钟。
所以排行榜上这一行的汇总值只代表这五栋；`compare` 只在两边都有（且输入相同）的建筑上重算汇总，和以后跑满十栋的结果比较时，没跑的五栋自动不参与。

- **时间上限**：每栋硬上限 3600 s，到 80%（2880 s）后不再开始新的上传请求。brandenburg-gate 只上传了 150 张里的 131 张，
  frauenkirche-dresden 只上传了 111 张里的 77 张（`summary.json` 的 `truncated = true`），结果是"用户上传到这里为止"的状态。
- **断点续跑**：五栋都是通过 `--resume` 复用的建筑运行（run key 相同：代码、配置、测试集、输入、选项）。四栋在
  2026-10-23 01:46–03:44 UTC 跑完，frauenkirche-dresden 在容器重启后 04:44–05:36 UTC 重跑。
- `commons-ten-v1/v0-noise.json`：真实 Commons 照片上的运行间噪声，只用最便宜的两栋（florence-baptistery、
  aachen-cathedral）各重复 3 次；它是噪声基线，不是性能记录。实测：洗礼堂 5 个可参评模型里每次通过 1 / 3 / 2 个
  （SfM 的细小差异让吸附落到不同的盆地），所以单栋建筑 `style_registration_passed` 差 2 个以内不算变化。
  这次重复时另一个工作流在同一台机器上占 CPU（负载 5–10），`wall_time_s` / `jobs_time_s` 的极差（约 590 s）被放大了，
  不代表正常的运行时间噪声。
- 作业目录（`sparse/0` + `images/`）、GLB、截图都留在仓库外的 `--work-root`；COLMAP 数据库已删掉。

## v1-after-fixes：窗口化吸附之后（metric_schema_version = 3）

`<testset>/v1-after-fixes.json` 是配准修复之后的记录：synthetic-v1、mdx-two-v1，以及 commons-ten-v1 的核心五栋（和 v0 一样，
其余五栋没跑）。三份都在 2026-10-23 10:06 起用同一份代码一次跑完（`provenance.code_sha256` 相同），SfM 也是重跑的。

改了什么（docs/STYLIZED_MODEL.md §3.2）：

- **吸附**（`crowd3d.stylize.snap_to_footprint`）：旋转只在 ±min(3σ_θ, P/2 − 5°) 里搜，σ_θ 是模型 JSON 的 `yaw_sigma_deg`
  （`reconstruct` 报的 GPS/罗盘融合 σ 再平方和加 3°，`Params.yaw_systematic_sigma_deg`），P 是 `outline_symmetry_period()`
  从轮廓量出的对称周期；窗口不到 2°（比萨斜塔这类近圆轮廓）只搜平移。旋转起点每 4° 一个；ICP 迭代里不截断平移，
  最终解让墙（吸附后落在轮廓 3 m 内的点）移动超过 25 m 就拒绝。接受看逐边核验，不看修正量小。
- **新的拒绝**（都是"宁可素模"）：最优解在窗口边上；歧义（±12 m 平移扫描里另有一个也过逐边核验、拟合差不多的位置：
  柱廊差一跨、门的另一面）；吸附后相机到墙的距离和照片自己的 EXIF GPS 不符（逐相机中位比偏离 1 超过 max(5%, 1.5 个标准误)）；
  轮廓要的尺度修正顶到 ±15%；单面模型的 GPS 尺度不确定度 > 15%。
- **新指标**：`style_registration_right`（核对过的通过里对的个数）、`style_rejected_placed_right`（被拒、但单靠配准就已经
  放对了的可参评模型：本该上色却成了素模，也算失败）、`style_snap_refused_gps`、`style_snap_refused_ambiguous`。
  加了指标，所以 `metric_schema_version` 从 2 升到 3，`compare v0-baseline v1-after-fixes` 要加 `--force`（两边共有的指标照常比较）。
- `HARNESS_VERSION` 4：harness 把模型 JSON 里的 σ_θ、尺度 σ 和每台相机自己的 GPS（`cameras[].gps_en`）交给吸附
  （`scripts/stylize_model.py snap_prior()`）。

**结果**（2026-10-23 10:06–12:49 UTC 跑完；`final` 判定和 v0 用同一个 harness 方法：通过的模型对照轮廓校正参考，
旋转 ≤ 5°、墙处 ≤ 3 m、尺度差 ≤ 15% 才算对）：

| 测试集 | 可参评 | 通过 | 通过率 | 对 | 错 | 被拒但配准本来就对 |
|---|---|---|---|---|---|---|
| commons-ten-v1 核心五栋 | 51 → 58 | 31 → 5 | 0.608 → 0.086 | 9 → 5 | 19 → 0 | – → 10 |
| synthetic-v1（生成器真值） | 12 → 12 | 6 → 4 | 0.50 → 0.33 | 4 → 4 | 2 → 0 | – → 4 |
| mdx-two-v1 | 2 → 2 | 1 → 1 | 0.50 → 0.50 | 1 → 1 | 0 → 0 | – → 0 |

commons 分建筑（v0 → v1）：

| 建筑 | 可参评 | 通过 | 对 | 错 | 被拒但配准本来就对 | 拒绝原因（v1） | 上传照片 |
|---|---|---|---|---|---|---|---|
| florence-baptistery | 6 → 5 | 4 → 0 | 2 → 0 | 0 → 0 | 0 | 尺度不确定 3、歧义 1、GPS 1 | 56 → 56 |
| aachen-cathedral | 2 → 2 | 1 → 0 | 0 → 0 | 0 → 0 | 0 | 超出 25 m 1、歧义 1 | 91 → 91 |
| brandenburg-gate | 16 → 22 | 9 → 0 | 1 → 0 | 8 → 0 | 7 | GPS 12、歧义 6、超出 25 m 2、超限缩放 1、尺度不确定 1 | 131 → 149 |
| leaning-tower-pisa | 12 → 11 | 8 → 1 | 0 → 1 | 8 → 0 | 2 | GPS 8、点太少 2 | 124 → 124 |
| frauenkirche-dresden | 15 → 18 | 9 → 4 | 6 → 4 | 3 → 0 | 1 | 歧义 6、点太少 2、窗口边 2、GPS 2、超出 25 m 1、残差 1 | 77 → 88 |

- **没有错误通过**：三个测试集 72 个可参评模型里通过 10 个，10 个都有参考可核对，都对（v0：38 个通过，可核对的 35 个里 21 个错）。
  commons 上 v1 的 5 个通过是比萨 W#27（只平移，墙处还差 1.25 m）和圣母教堂 SW#11/14/15/20（墙处 0.77–1.57 m）。
- **代价**：对的通过从 9 个降到 5 个，另有 10 个模型被拒、但单靠配准就已经放对了（勃兰登堡门 7、比萨 2、圣母教堂 1：
  勃兰登堡门 E 立面有 4 个模型的墙点（中位）落在 OSM 东边线外 5.4–6.6 m，另 2 个贴着边线、但 13–14 m 外另有一个拟合得一样好的
  位置（柱廊），都被 GPS 核对或歧义拒掉；比萨 W 立面重建出的弧太平，吸附后相机比 GPS 近 11%）。
  这些按任务的口径也算失败：本该上色的地方是素模。
- **用户看到的**：每个扇区显示**最新**发布的模型。v1 里五栋的最新模型都没过门（圣母教堂 SW 扇区通过的 #11/#14/#15/#20
  上了 1–4 面墙，但被后来的 SW#30 替换，它因为歧义被拒；比萨 W#27 通过但被 W#36 替换，W#27 自己也 0 面墙过了逐墙检查），
  所以 `style_walls_coloured` 五栋都是 0（v0 是 16 面，其中 3 栋的展示模型是错误通过）。"新模型没过门时继续显示上一个过门的模型"
  是要负责人决定的产品规则，见 STYLIZED_MODEL §5.2。
- **可比性**：SfM 是重跑的，模型集合不完全一样（勃兰登堡门 16 → 22 个可参评）。v0 运行时另一个工作流占着 CPU，
  勃兰登堡门和圣母教堂撞到时间上限，只上传了 131/150、77/111 张；v1 机器空闲，上传了 149、88 张（仍然 `truncated = true`）。
  `compare` 认为两边输入相同（列出的照片一样），所以汇总照常比较，但这部分差异来自上传量，不是代码。
- **噪声**：v0-noise 的极差（单栋 `style_registration_passed` 2 个、通过率 0.4）仍然适用；v1 的模型级结果在不同 SfM 运行之间
  也会翻（例如洗礼堂 E#14 在本轮之前的一次 v1 运行里通过且对，这次 GPS 比 1.09 超过容差 7.8%、又有一个 4.5° 外的歧义解，被拒）。
- 诚实 QA、`audit_model`、`audit_glb`：三个测试集 76 个风格化模型全部通过，0 违规。
