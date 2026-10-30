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
| `buildings.<slug>` | `status`（ok / failed / timeout / skipped）、`metrics`（全部指标；`--repeat` 时是各次运行的中位数，第 1 次的原值在 `metrics_first`，`details`/`artefacts` 也属于第 1 次）、`inputs_sha256`（实际用到的照片字节、轮廓、参考模型的指纹）、`details`（漏斗、扇区、作业、每个模型的质量门原因、轮廓拟合、GT 误差；指标第 7 版起每个模型还有 `geo`（和参考比的地理误差、偏航和尺度的 z）和 `coverage`（它支持的轮廓米数），外加整栋的 `coverage` 分箱参数）、`artefacts`（仓库外的路径）；`--repeat` 时还有每次的 `repeats` |
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

## v1-after-fixes：第三轮配准修复之后（metric_schema_version = 5）

`<testset>/v1-after-fixes.json` 是第三轮配准修复之后的记录：synthetic-v1、mdx-two-v1，以及 commons-ten-v1 的核心五栋（和 v0
一样，其余五栋没跑：两次重复的核心五栋就要约 5.5 h）。每栋**重复 2 次**（`--repeat 2`，SfM 也重跑），记录的指标是两次的中位数
（计数可能是 x.5），`noise` 块是两次的极差。三份都用同一份代码（`provenance.code_sha256` = 5e84f665…），2026-10-24
00:16–05:54 UTC 跑完。第二轮的记录（schema 4）被覆盖了，它在提交 3bbb0dd 里。

**为什么有第三轮**：验证者指出第二轮仅有的 3 个真实通过（圣母教堂 SW#11/#14/#15）里，SW#14、#15 的相机离参考 2.2 / 2.3 m
（用户的定义是相机中心误差），SW#14 的贴图按参考位姿重投影整体偏高 3–4 m（模型倾斜 2.9°，二维检查看不见）；对抗用例里两面墙的
L 形模型尺度错 7% 照样通过，邻楼立面能被滑到这面墙上；被标记的相机照样给墙上色；机器忙时风格化会被上传时间上限整栋跳过。

**改了什么**（docs/STYLIZED_MODEL.md §3.2）：

- 判定（schema 5）：相机水平中心离参考 ≤ 1.5 m（中位数）**并且**转角 ≤ 5°、墙点挪动的 90 分位 ≤ 1.5 m（三维，旋转来自相机
  朝向）、尺度差 ≤ 10%。`style_ref_camera_error_worst_m` 改名 `style_ref_camera_median_worst_m`；新增
  `style_ref_camera_error_max_m`、`style_ref_walls_p90_worst_m`、`style_incomplete`。
- 门：每个歧义检查都把相机的位移也算上（转角谱另试"让相机挪 1.5 m 的转角"）；尺度谱（绕墙缩放 ±3/6/10%，在 GPS 尺度 σ 的
  2 倍以内）；按相机高度的倾斜检查；有墙点却没有相机看得见的面直接拒绝；σ_θ 在风格化里也有 3° 的下限；配准过的模型必须带相机。
- 被标记的相机不参与上色；风格化有自己的时间预算（1800 s）。

**结果**（两次重复给出的数完全一样的写一个数，不一样的写 r1 / r2）：

| 测试集 | 可参评 | 通过（v0 → 第二轮 → 第三轮） | 对 | 错 | 被拒但配准本来就对 | 不可参评却过门 |
|---|---|---|---|---|---|---|
| commons-ten-v1 核心五栋 | 59 / 60 | 31 → 3 → **0** | 0 | 0 | 0 | 0 |
| synthetic-v1（生成器真值） | 12 | 6 → 3 → 2 | 2 | 0 | 2（尺度没定住 1、只拍到一面墙 1） | 0 |
| mdx-two-v1 | 2 | 1 → 0 → 0 | 0 | 0 | 0 | 0 |

`style_registration_pass_rate`（v0 → 第二轮 → 第三轮）：

| 建筑 | v0 | 第二轮 | 第三轮 | 第三轮拒绝原因（两次合计） | 上传照片 |
|---|---|---|---|---|---|
| florence-baptistery | 0.667（4/6） | 0（0/5.5） | 0（0/5.5） | 贴得太松 4、歧义 2、偏航谷平 2、窗口边 2、超限缩放 1 | 56/56 |
| aachen-cathedral | 0.5（1/2） | 0（0/2） | 0（0/2） | 超限缩放 2、窗口边 2 | 91/91 |
| brandenburg-gate | 0.5625（9/16） | 0（0/16） | 0（0/22） | 歧义 24、超限缩放 10、偏航谷平 6、贴得太松 2、超出 25 m 2 | 149/150 |
| leaning-tower-pisa | 0.667（8/12） | 0（0/11.5） | 0（0/12） | 偏航查不了 7、尺度没定住 7、贴得太松 6、点太少 4 | 124/124 |
| frauenkirche-dresden | 0.6（9/15） | 0.167（3/18） | 0（0/18） | 贴得太松 19、偏航谷平 7、点太少 4、超出 25 m 2、超限缩放 1、歧义 1、残差 1、窗口边 1 | 88/111 |
| **合计** | 0.608 | 0.0566 | **0** | | |

- **第二轮的 3 个通过**：SW#11 被"转 −1.2° 相机就挪 1.5 m、拟合仍在余量内"拒（它按 schema 5 本来是对的：相机 0.93 m、三维墙点
  1.31 m），SW#14、#15 被相机位移的歧义拒（倾斜检查也会拒：推出 3.04 / 2.55 m 的竖直偏移）。按 schema 5 重判第二轮的吸附：SW#14
  错（三维墙点 3.78 m、相机 2.2 m），SW#15 错（相机 2.33 m）。
- 119 个拒绝里 24 个只因为相机那一条成立（照片在墙上挪 < 1.5 m，相机却挪 ≥ 1.5 m）。按 schema 5，没有一个 commons 模型单靠配准
  就放对了（`style_rejected_placed_right` = 0），所以这次通过率掉到 0 不是"把对的拒了"，而是相机标准下本来就没有对的。
- **用户看到的**：五栋都是 0 面上色墙（v0 16 面，其中勃兰登堡门、比萨、圣母教堂的展示模型是错误通过）。
- 诚实 QA、`audit_model`、`audit_glb`：三个测试集全部 156 个风格化模型都通过，0 违规；`style_incomplete` = 0；commons 有 185 台次
  被标记的相机没参与上色。
- **可比性**：机器比第二轮空，勃兰登堡门上传了 149/150 张（第二轮 131），可参评模型因此更多；勃兰登堡门、圣母教堂仍撞到时间上限
  （`truncated = true`）。离线重放、对抗用例和变异检查见 STYLIZED_MODEL §5.2。

## v1-merged：合并结构优先分支之后（metric_schema_version = 5）

`synthetic-v1/v1-merged.json`、`mdx-two-v1/v1-merged.json` 是合并后的主线（结构优先分支连同 D24 在 c3416ce 合并，加合并修复第二轮，
提交 0a49b22，工作区干净，`provenance.code_sha256` = 96d3b9f9…）在默认路径上的记录：和 v1-after-fixes 同样的命令、配置和
`--repeat 2`，2026-10-24 22:13–22:20 UTC。默认路径不用结构体块（基准和 `scripts/stylize_model.py` 都不传 `SceneInput.structure`），
配准门和 v1-after-fixes 相同。commons-ten-v1 没有重录（两次重复的核心五栋要约 5.5 h），那里仍是 v1-after-fixes：它的 59.5 个
可参评模型一个都没通过配准门，没有墙上色，所以下面这些只影响上色结果的改动在那份记录里本来就显不出来。

**和 v1-after-fixes 比**（`benchmark.py compare`）：

- synthetic-v1：所有重建、配准和 `style_registration_*` / 吸附指标相同或在噪声内（通过 2/12，对 2、错 0，被拒但本来放对了 2，
  上色墙 6 面）。变的只有上色：`style_observed_area` 0.579 → 0.567（synthetic-two-sides 0.419 → 0.392，`compare` 判为显著；
  synthetic-ring 0.739 → 0.742），`style_roof_observed_area` 0.01 → 0（ring 0.02 → 0），`style_fill_m2` 0.336 → 0.438（在记录的
  极差 0.11 内），`style_glb_kb_gz` 21.05 → 20.9。这些来自分支自己的默认路径改动，不是合并造成的：合并核查（merge-sf verify-r1）
  用同一个吸附重新风格化，分支代码（9c378ae）和合并后的代码给出逐字节相同的 GLB；把合并后代码的颜色链默认值（中性白平衡锚点、
  逐墙白平衡回锚、条纹、`roof_lift` 0.2 → 0.5、`relief_detail`）和 `roof_min_cameras`（1 → 0）改回去，就得到 v1-after-fixes 的
  掩码和屋顶份额。中性锚点改变了天空 / 植被掩码；没有相机正对的平屋顶不再算"观测到"。
- mdx-two-v1：两个可参评模型两次都被吸附歧义拒绝，0 面上色墙，和 v1-after-fixes 一样。重建指标变了：亚琛两次都发布了 1 个模型
  （v1-after-fixes 是一次 1 个、一次 0 个），它的轮廓转角 50.1° 让汇总的 `outline_rotation_abs_deg` 从 1.87 变成 26.4（`compare`
  判为显著）；洗礼堂的轮廓转角 1.87 → 2.65°。这些是重建在不同运行之间的差异：从 v1-after-fixes 的代码（1d6055b）到 0a49b22，
  改动的只有 `crowd3d/stylize.py` 和新的结构模块（judge、lines3d、relief、skyline、structmetrics、structure）以及
  `scripts/stylize_model.py`，重建和评测代码一行没动；`default-noise` 里亚琛的轮廓转角极差是 45.7°。每次 2 次重复会低估这种
  噪声（docs/BENCHMARK.md §8）。运行时间（125.5 → 63.5 s）不能跨两次会话比。

## 外部评审采纳计划：B0、F 和拆分（metric_schema_version = 7）

方法、表和解读在 [docs/REVIEW_ADOPTION.md](../../docs/REVIEW_ADOPTION.md)。这些文件是当时写出的结果原样拷进来的（逐字节相同，所以补算结果里 `provenance.rescored_from.sha256` 对得上），文件名就是运行名。

| 文件 | 是什么 |
|---|---|
| `synthetic-v1/b0.json`、`mdx-two-v1/b0.json` | 基线 B0：819cb78 的产品代码新跑，2 次重复（测试流程有一行没提交的补丁：相似变换尺度 ≤ 0 时不算尺度误差，后来在 257fdf0 提交） |
| `synthetic-v1/v1-merged-s7.json`、`mdx-two-v1/v1-merged-s7.json`、`commons-ten-v1/v1-after-fixes-s7.json` | v1-merged、v1-after-fixes 的产物在 819cb78 上用 `rescore --restylize` 补算到第 7 版；核心五栋的 B0 就是 `v1-after-fixes-s7` |
| `commons-ten-v1/b0-dev.json` | B0 新跑洗礼堂、亚琛各 1 次（看运行时间） |
| `commons-ten-v1/b0-test.json` | B0 新跑留出五栋，2 次重复，留出集上第一次跑 SfM |
| `synthetic-v1/f.json`、`mdx-two-v1/f.json`、`commons-ten-v1/f-dev.json`、`commons-ten-v1/f-test.json` | 冻结候选 F：5e44646 的全部默认值，2 次重复；`f-dev` 的勃兰登堡门、圣母教堂按 v1-after-fixes 的请求数截断（104、31）；`f-test` 是留出集上第二次、也是最后一次 SfM |
| `commons-ten-v1/f-uncapped.json` | F，勃兰登堡门和圣母教堂不设请求上限、14,400 s 上限，各 1 次（吞吐量） |
| `<测试集>/ablation-v5/<b0\|f>[-dev\|-test]-r<P1><P3>.json` | 2×2×2 拆分：5e44646 的 `rescore --restylize`，P1 = `--snap-gate on\|off`（B0 的产物开时加 `--regeo`），P3 = `--structure on\|off`。放在子目录里，不进排行榜，也不进 `compare` 自动收集的噪声来源 |

- 比较时 `compare` 会把同一测试集目录里所有同版本结果的噪声块都算进门槛，dev 和 test 的结果在同一目录里时另一个划分的汇总极差也会混进来；要复现 REVIEW_ADOPTION 里的数，加 `--results-dir <空目录>`（docs/BENCHMARK.md §8）。
- 风格化指标：B0 的 `v1-*-s7` 是 819cb78 的风格化（按轮廓拉伸的体块），F 的是结构流水线（OSM 分件体块），`style_walls_total`、`style_glb_kb_gz`、`style_time_s` 的含义不同；只比风格化时用 `ablation-v5/` 里同一份代码打分的格子。
