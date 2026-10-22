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
