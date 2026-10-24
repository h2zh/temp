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

## 风格化指标（metric_schema_version = 2 起；现行 5）

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
  通过的模型在吸附之后还需要一个相似变换（吸附后的相机 → 参考相机）。metric_schema_version 5 起，四条都满足才算对，否则记为
  "错误通过"（`style_registration_wrong`，没满足的几条记在 `reference.wrong_because`）：吸附后相机水平中心离参考的中位数
  ≤ 1.5 m；二维相似变换转角 ≤ 5°；尺度差 ≤ 10%；模型墙点（吸附后落在轮廓 3 m 内的墙高点）还要挪动的 90 分位 ≤ 1.5 m，
  取二维和三维（旋转来自相机朝向，所以倾斜也算）里大的那个。`style_ref_camera_median_worst_m` 是通过模型里最大的相机中位误差
  （schema 4 叫 `style_ref_camera_error_worst_m`），`style_ref_camera_error_max_m` 是最差的单台相机，
  `style_ref_walls_p90_worst_m` 是最大的墙点位移。schema 4 只在墙上判（二维），没有相机那条：圣母教堂 SW#14、#15 因此算对，
  按 schema 5 是错的（SW#14 倾斜 2.8°，贴图整体偏高约 3.5 m）。schema 3 及以前是"旋转 ≤ 5°、墙点中位处 ≤ 3 m、尺度差 ≤ 15%"。
  少于 3 台相机有参考位姿时不下结论。参考自己的误差：圣母教堂参考的偏航大约差 1–2°、尺度差几个百分点（相机离墙 80 m 时相当于
  1.5–3 m），所以 1–2 m 量级的判定两个方向都可能错；近圆轮廓（比萨斜塔）上参考模型自己的旋转也定不下来；合成集用的是生成器
  真值，没有这个问题。
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
  平均距离 > 0.45 m，轮廓定不住偏航）；`ambiguous` 也包括"偏航谷太平"（绕墙转 5° 仍拟合得一样好）。schema 5 再加 `tilt`
  （相机高度说明模型是斜的，或相机太少、分布太窄而查不了）、`unseen_face`（吸附后某条边上有墙点，却没有一台相机能隔着建筑
  看到它们）、`no_cameras`（标了配准却没有相机位置）；`ambiguous` 还包括把相机挪 ≥ 1.5 m 的竞争解和"缩放 ±3/6/10% 仍拟合得
  一样好"（尺度没定住）。
- 每栋建筑的"最佳模型"（用户最多能看到的）：最新发布的模型里上色墙最多的那个，其次看观测面积；`style_walls_*`、
  `style_observed_area`、`style_fill_m2`、`style_glb_kb_gz` 都描述它。墙的状态按 §3.3：已解锁 = 掩码 ≥ 60% 且 ≥ 3 台相机
  各看到 ≥ 15%；其余上了色的是部分解锁；其他是素模（待解锁）。
- `style_qa_failed`、`style_provenance_violations` 必须为 0（诚实 QA、`audit_model`、`audit_glb`）。
- **风格化有自己的时间预算**（harness 5）：从风格化开始算 `STYLE_TIME_CAP_S` = 1800 s，上传的时间上限不再吃掉它（以前从运行
  开始算 1.15 倍软上限，机器忙时圣母教堂整栋的风格化记录都被跳过）。没来得及做的模型记在 `style_incomplete`；只要一边有，
  `compare` 就把这栋的 style_* 指标标成 `incomplete`，不算变化，汇总也不算这栋。
- 被标记的相机（和自己的 GPS / 罗盘不符、出了半径、焦距退化）不参与上色（harness 5，`load_job`）。

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

## v1-after-fixes：第二轮配准修复之后（metric_schema_version = 4）

`<testset>/v1-after-fixes.json` 是第二轮配准修复之后的记录：synthetic-v1、mdx-two-v1，以及 commons-ten-v1 的核心五栋（和 v0
一样，其余五栋没跑）。每栋**重复 2 次**（`--repeat 2`，SfM 也重跑）：记录的指标是两次的中位数（两个数取平均，所以计数可能是
x.5），`noise` 块是两次的极差。三份都用同一份代码（`provenance.code_sha256` = f4e97cf0…），2026-10-23 15:58–21:35 UTC 跑完。
第一轮的 v1 记录（schema 3）被覆盖了，它在提交 60bc217 里。

**为什么有第二轮**：第一轮报"0 个错误通过"，验证者用同一份代码重跑圣母教堂就出了一个错误通过（SW#24，上了 12 面墙，比参考
多转 3.3°）；`prior_only` 的 SW#17 照样过门上色；近圆轮廓只搜平移时任何偏航误差都能过（对抗用例里 47 个通过有 34 个错）；
第一轮"对"的判定（墙点中位处 ≤ 3 m、尺度 ≤ 15%）也太松。

**改了什么**（docs/STYLIZED_MODEL.md §3.2）：

- `prior_only` 的模型（模型 JSON 没写 `georegistered: true`）不吸附、不过门，整栋素模；`style_gate_passed_not_eligible` 必须为 0。
- 粗搜索改成固定网格（窗口内约每 2° 一个转角 × 平移 ±25 m 每 1.5 m），ICP 只用离轮廓 ≤ max(1 m, 3 × 中位距离) 的点对；
  候选解的刚体版和（轮廓能修尺度时的）缩放版一起按截断平均距离排序、一起比。
- 歧义：另一个过逐边核验、拟合差不多的解会把照片在墙上挪 ≥ 1.5 m（90 分位）或转 ≥ 2°；或者最优解绕墙转 2 / 3.5 / 5°（平移重拟合）
  仍拟合得差不多、照片挪 ≥ 1.5 m，或转 5° 也还差不多（偏航谷太平）。
- 墙点到轮廓的平均距离（截断 3 m）> 0.45 m 也拒绝（贴得太松，定不住偏航；只在圣母教堂 7 个模型上标定过）。
- 近圆轮廓只搜平移时，3σ_θ > 5° 或让照片绕轮廓挪 > 1.5 m 就拒绝；按 3° 的系统误差底，比萨斜塔因此一律素模。
- 判定（schema 4）：转角 ≤ 5°、墙点挪动的 90 分位 ≤ 1.5 m、尺度差 ≤ 10% 才算对；相机中心误差另记，不进判定。新指标
  `style_gate_passed_not_eligible`、`style_ref_camera_error_worst_m`。参考本身没变（参考的轮廓拟合仍用原来的 70% 截断）。

**结果**（两次重复给出的数完全一样的写一个数，不一样的写 r1 / r2）：

| 测试集 | 可参评 | 通过 | 通过率 | 对 | 错 | 被拒但配准本来就对 | 不可参评却过门 |
|---|---|---|---|---|---|---|---|
| commons-ten-v1 核心五栋 | 51 → 53 | 31 → 3 | 0.608 → 0.057 | 3 | 0（v0 按 schema 2 判 19 个错；按 schema 4 重判 23 个错、5 个对） | 0 | 0（v0 有 1 个：SW#17） |
| synthetic-v1（生成器真值） | 12 → 12 | 6 → 3 | 0.50 → 0.25 | 3 | 0（v0：2 个错；按 schema 4 重判 3 个错） | 2（都是只拍到一面墙的模型） | 0 |
| mdx-two-v1 | 2 → 2 | 1 → 0 | 0.50 → 0 | 0 | 0（v0 的 E#4 按 schema 4 重判是错的：墙点 90 分位还差 1.63 m） | 0 | 0 |

commons 分建筑（v0 → v1 r1 / r2；v0 的"对/错"按 schema 4 重判，括号里是 v0 记录里 schema 2 的判定）：

| 建筑 | 可参评 | 通过 | 对 | 错 | 拒绝原因（v1，两次相同处只写一次） | 上传照片 |
|---|---|---|---|---|---|---|
| florence-baptistery | 6 → 6 / 5 | 4 → 0 | 2 → 0 | 0 (0) → 0 | 贴得太松 2、歧义 2 / 1、偏航谷平 1、窗口边 1 | 56 → 56 |
| aachen-cathedral | 2 → 2 | 1 → 0 | 0 → 0 | 0 (0) → 0 | 超限缩放 1、窗口边 1 | 91 → 91 |
| brandenburg-gate | 16 → 16 | 9 → 0 | 0 → 0 | 9 (8) → 0 | 歧义 10、超限缩放 4、贴得太松 1、超出 25 m 1 | 131 → 131 |
| leaning-tower-pisa | 12 → 11 / 12 | 8 → 0 | 0 → 0 | 8 (8) → 0 | 偏航查不了 6、贴得太松 3 / 4、点太少 2 | 124 → 124 |
| frauenkirche-dresden | 15 → 18 | 9 → 3 | 3 → 3 | 6 (3) → 0 | 贴得太松 10、点太少 2、超限缩放 1、超出 25 m 1、尺度不确定 1 | 77 → 88 |

- **通过的 3 个**：圣母教堂 SW#11 / #14 / #15（两次都是这三个）。吸附后还差 −0.1 / −0.8 / 1.6°、墙点 90 分位 1.02 / 1.17 / 0.93 m、
  尺度 0.4 / 1.9 / 2.3%；相机中心误差中位 0.93 / 2.2 / 2.3 m、最大 1.3 / 2.9 / 13.0 m（SW#15 有一台相机差 13 m，是它自己的位姿，
  吸附只能整体移动）。分别上色 1 / 3 / 1–2 面墙。
- **用户看到的**：每个扇区显示**最新**发布的模型；五栋的最新模型都没过门，所以 `style_walls_coloured` 五栋都是 0（v0 是 16 面，
  其中勃兰登堡门、比萨、圣母教堂的展示模型是错误通过）。"新模型没过门时继续显示上一个过门的模型"仍是要负责人定的产品规则；
  第一轮拿来举例的 SW#20 按 schema 4 是错的（还要 1.14 倍尺度），现在也不过门。
- **噪声**：两次重复之间，可参评数在洗礼堂、比萨差 1（SfM 发布的模型数不同），通过、对、错都没有差别；第一轮"0 个错"不能
  复现的原因（SW#24 的竞争解差 1.6°、落在旧的 2° 门槛内）由偏航谷检查和"贴得太松"兜住了，离线重放见 STYLIZED_MODEL §5.2。
- **可比性**：v1 运行时另一个工作流占了约 2.5 个核（负载 4–6）：勃兰登堡门、圣母教堂撞到时间上限（`truncated = true`），
  只上传了 131/150、88/111 张；亚琛每次 690–705 s（v0 636 s）。`wall_time_s` 的差异主要来自这里，不是代码。
- 诚实 QA、`audit_model`、`audit_glb`：三个测试集全部风格化模型都通过，0 违规（`style_qa_failed`、`style_provenance_violations` = 0）。
