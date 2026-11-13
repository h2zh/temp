# 决定的索引

产品负责人做过的决定，按编号排。每行是编号、日期、一句话，和记录在哪。要看决定的原话、开发这边的理解、做了什么、怎么验证的，点进记录。

D30–D33 的记录原来是 README 里的四节，D34 C 部分第 2 步（2026-11-13）把它们搬成了这个目录里的文件，文字没有改。更早的决定记在各自的设计文档里，没有搬，这里只给链接。

| 编号 | 日期 | 一句话 | 记录 |
|---|---|---|---|
| （没有编号） | 2026-10-22 | 视觉定位的四条约定：没有 GPS 的照片仍要有 EXIF；这种照片在地标页上明确选地标；靠视觉定位加入的贡献不记打卡；地标只能从 Wikidata 条目建 | [VISUAL_LOCALIZATION.md §2.5](../VISUAL_LOCALIZATION.md#25-已确认的决定修改提议与开放问题) |
| D1–D14 | — | 风格化模型的设计交给负责人定的 14 件事。负责人拍了板的：D7（画风由负责人定，做成可替换的文件）、D8（接受"模型是玩具，不是真实记录"）、D12（允许补墙上的小洞，后来被 D29 撤销）。D1、D2、D6、D14 在 D29 之后作废，其余按文档建议的默认值 | [STYLIZED_MODEL.md §9.2](../STYLIZED_MODEL.md#92-需要用户决定的事项) |
| D15 | 2026-10-23 | 至少两路照片证据一致，就自动采用照片值，改掉地图上的值 | [STRUCTURE_FIRST.md 第 6 节](../STRUCTURE_FIRST.md#6-需要您决定的事) |
| D16–D23 | — | 结构优先的设计里别的待定事项（整块参数带动没拍到的面、只有屋顶形状时的约定比例、引导拍摄、政府的三维数据、新参数、颜色取向、馆员录入、以盲测为准上线）。都按文档建议的默认值做，负责人没有逐条确认；D20 后来被 D25 改了一处 | [STRUCTURE_FIRST.md 第 6 节](../STRUCTURE_FIRST.md#6-需要您决定的事) |
| D24 | 2026-10-24 | 只凭照片的点量出的立面凹凸可以建，但先补上"合并开间"的漏洞再打开 | [STRUCTURE_FIRST.md 第 6 节](../STRUCTURE_FIRST.md#6-需要您决定的事)、[§2.4.7](../STRUCTURE_FIRST.md#247-照片实测的结构细节已实现crowd3dreliefpycrowd3dstructmetricspyd24-已打开) |
| D25 | 2026-10-25 | 方向 A：证据很强时，照片直接纠正 OSM（高度、倾斜、缺高度的部分补出体块），每处可以撤销 | [STRUCTURE_FIRST.md §2.4.8](../STRUCTURE_FIRST.md#248-强证据纠正-osmd25方向-a产品负责人-2026-10-25-拍板d26-以后只往上改) |
| D26 | 2026-10-28 | 放宽标准，清掉只为对抗场景和伪造记录加的代码；D25 不再往下改高度 | [STRUCTURE_FIRST.md 第 6 节](../STRUCTURE_FIRST.md#6-需要您决定的事) |
| D27 | 2026-10-28 | D15 改高度时，要读到这块积木自己的屋顶 | [STRUCTURE_FIRST.md 第 6 节](../STRUCTURE_FIRST.md#6-需要您决定的事) |
| D28 | — | 还没有决定。这个编号留给"只有一种可商用的天空掩码时怎么办" | [STATUS.md"下一步"第 2 条](../STATUS.md#下一步) |
| （没有编号） | 2026-10-29 | 采纳外部评审的四个包（地理锚点、持久地图、结构优先接进产品、留出集评测），并在测试集上验证 | [REVIEW_ADOPTION.md](../REVIEW_ADOPTION.md) |
| D29 | 2026-11-03 | 放弃给模型上照片颜色，删掉上色代码；模型一律是素模 | [STRUCTURE_FIRST.md 第 6 节](../STRUCTURE_FIRST.md#6-需要您决定的事)、[STYLIZED_MODEL.md §9.2](../STYLIZED_MODEL.md#92-需要用户决定的事项) |
| D30 | 2026-11-03（11-04 验证） | 别人照片的拍摄位置取整到 1 米、1°，不带用户 id；只显示同意公开的；视觉定位做了，代价超标，默认关 | [D30-photo-poses.md](D30-photo-poses.md) |
| D31 | 2026-11-03 | 照片资格（建筑是主体，或者相机在建筑 50 m 以内）和立面按注册张数过门；全新运行按事先写好的标准是 FAIL，留不留等负责人定 | [D31-photo-eligibility.md](D31-photo-eligibility.md) |
| D32 | 2026-11-05 | 上传必须同意用户协议（拍摄位置和照片公开）；照片和素模对照；浅色 / 深色和中英文切换 | [D32-terms-compare-appearance.md](D32-terms-compare-appearance.md) |
| D33 | 2026-11-05 | 用户协议在登记时同意一次；视角取整到 1° | [D33-registration-fov.md](D33-registration-fov.md) |
| D34 | 2026-11-10 | 按代码评审的建议修改，原图照旧保存。A 部分是安全和隐私，B 部分是结果的正确性，C 部分是工程效率 | A、B：[D34-review-fixes.md](D34-review-fixes.md)；C：[D34-part-c-engineering.md](D34-part-c-engineering.md) |

还在等负责人答复的问题（D30、D31、D33 各留了几个）列在 [STATUS.md 的"下一步"](../STATUS.md#下一步)。

## 数字出处和 scratch 路径

各份文档给结论的时候，常常写上产物在哪：日志、中间结果、对比图、一次性的驱动脚本。这些东西大多不在仓库里，在开发这个原型的那次会话的临时目录里。文档里统一把这个目录写成 `<scratch>`，路径写成 `<scratch>/…`。

- **`<scratch>` 是什么。** 开发机上的 `/tmp/claude-0/-home-user-temp/04ee4750-a5e3-5a4d-b130-9dfde89f5621/scratchpad`。它不是仓库里的路径，别的机器上也没有。
- **它不能从仓库重建。** 那是容器里的临时目录，容器回收就没了。里面有照片衍生的东西（点云、GLB、截图、天空掩码），按许可本来就不能进仓库；也有不含照片的东西（笔记、日志、JSON、脚本），只是当时没有提交。
- **所以 `<scratch>/…` 要读作"当时在那里"**，不是"你能在那里找到"。能从仓库重算的结果，文档另外给了命令（[REVIEW_ADOPTION.md](../REVIEW_ADOPTION.md) 第 7 节、[BENCHMARK.md](../BENCHMARK.md) 的 `rescore`）；别的要重新跑，有的还要先把驱动脚本重写出来（见下）。
- **D34 的两份记录**里写的"开发机的草稿目录""草稿目录"，指的是同一个目录。
- 2026-11-13（D34 C 部分第 2 步）以前，六份文档里写的是完整的绝对路径或者它的缩写，看起来像仓库里的路径。那一步把 120 处改成了 `<scratch>/…`（[STRUCTURE_FIRST.md](../STRUCTURE_FIRST.md) 103 处，[VISUAL_LOCALIZATION.md](../VISUAL_LOCALIZATION.md) 7 处，[REVIEW_ADOPTION.md](../REVIEW_ADOPTION.md) 4 处，[REAL_WORLD_EVAL.md](../REAL_WORLD_EVAL.md) 3 处，[STYLIZED_MODEL.md](../STYLIZED_MODEL.md) 2 处，[ART_DIRECTION.md](../ART_DIRECTION.md) 1 处），指向的东西一处都没有删。`tests/test_docs.py` 守着：除了本节，任何文档里都不许再出现这个目录的绝对路径。

**代码和数据里还留着这个目录的地方**（没有改）：

- `scripts/benchmark.py` 的 `_SCRATCH`：没有设 `$CROWD3D_TESTSET_ROOT`、`$CROWD3D_BENCH_DIR` 时，`--data-root` 和 `--work-root` 的默认值是这个目录下的 `ten/testsets` 和 `ten/bench`。在别的机器上要设这两个环境变量，或者每次传参数（[BENCHMARK.md](../BENCHMARK.md) 第 3 节）。
- `eval/results/` 下 102 个结果文件里有 5,741 处这个目录的绝对路径：运行的命令行，每栋建筑的数据目录、日志，每个模型的 GLB 在哪。它们是运行当时的记录，不改。`rescore` 和 `--resume` 要用的产物如果已经不在，只能从头跑。

**结果来自不在仓库里的驱动脚本的地方**（数字都留在文档里，引用它们的句子旁边标了这一点）：

| 结果 | 量它的驱动 | 仓库里有什么，缺什么 |
|---|---|---|
| D25-T：比萨斜塔的倾斜 4.59°、朝 177.0°，以及 11 个众包重建上的结果（[STRUCTURE_FIRST.md](../STRUCTURE_FIRST.md) §2.4.8 第十四部分，[STATUS.md](../STATUS.md)） | `<scratch>/dir-a/` 下的 `pisa_tilt.py`（另用了同一个目录里的三个读取模块）；D26 之后的重跑是 `<scratch>/cleanup/step6/tools/pisa_rule_t.py` | 规则和测量在仓库里：`crowd3d/tilt.py`，`crowd3d/correct.py` 的 `rule_t`、`tilt_consensus`、`apply_tilt`。但**仓库里没有任何入口调用它们**，产品、`structure_pipeline` 和脚本都不调，只有单元测试调。所以这些数从仓库复现不了，缺的是驱动脚本。这个入口还没有补，列在 [STATUS.md](../STATUS.md) |
| D24、D25 在洗礼堂和亚琛的参考场景上的结果（85 个数字里 36 个来自照片；唱诗堂顶高"至少 36.26 m"）（[STRUCTURE_FIRST.md](../STRUCTURE_FIRST.md)，[STATUS.md](../STATUS.md)） | `<scratch>/cleanup/step6/tools/run_ref_step6.py`，用了研究用的第二种天空掩码 Dm+R | 规则和流水线在仓库里（`scripts/stylize_model.py --scene`、`crowd3d/structure_pipeline.py`）。Dm+R 掩码的许可没有确认可以商用，`crowd3d` 只认 A+R，所以这组数用仓库里的代码跑不出来：只用 A+R 时这两个参考场景上照片数字都是 0 |
| D31、D30 的通过标准里"要读运行产物的核对"（`d31-v1` 六项、`d30-v1` 九项）（[BENCHMARK.md](../BENCHMARK.md) 6.11、6.12 节） | 读运行目录里的产物，由人按标准文件写的方法判；`compare` 只列出名字 | 判法写在 `eval/criteria/*.json` 里，判的过程写在 BENCHMARK.md 里，没有可以重跑的代码 |
| D34 第 6、7 步的判定、裁决计数和单个作业的重跑（[BENCHMARK.md](../BENCHMARK.md) 6.13、6.14 节，[D34-review-fixes.md](D34-review-fixes.md)） | 草稿目录里的几个小脚本 | 规则的算法和每一步用到的数写在标准文件里；脚本没有进仓库 |
