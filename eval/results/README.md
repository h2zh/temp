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
  通过的模型在吸附之后还需要的修正（相机中心之间的相似变换）若旋转超过 5°、在墙所在位置移动超过 3 m 或尺度差超过 15%，
  记为"错误通过"（`style_registration_wrong`）。少于 3 台相机有参考位姿时不下结论。
  这个判定有自己的误差：近似圆形的轮廓（比萨斜塔）上参考模型自己的旋转也定不下来（它的轮廓拟合转了 3.2°），
  从相机外推到墙的位置在真实参考上可差约 4 m。所以真实集上略超门槛的"错误"（旋转 < 8°、墙处 < 6 m、尺度差 < 20%）
  只算"偏离"，超过这些的才是确定的错误；合成集用的是生成器真值，没有这个问题。
- **失败原因**（每个未通过的可参评模型，`details.models[].style.diagnosis.cause`）：`single_wall`（只有一条边有点，或通过的边
  全平行）、`too_few_points`、`symmetric_alias`（不加约束的 ICP 收敛到离正确修正 > 10° 的解）、`georegistration`（正确的修正
  本身就超出吸附上限 20° / 20 m）、`osm_outline`（参考模型自己都贴不上轮廓）、`residual`（方向对了但墙仍离轮廓太远）。
  证据优先用参考模型；没有参考时用 harness 自己的轮廓拟合（`outline_check`，对称建筑上只是下限）。
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
