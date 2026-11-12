# D34 C 部分：工程效率（2026-11-12 起）

D34 的决定和前两部分（A：安全和隐私；B：压缩、结构的错误、地理配准）记在 [D34-review-fixes.md](D34-review-fixes.md)。本文记 C 部分：评审"工程做法"那一块的发现，一步一节。

**C 部分只有一条规矩：产品算出来的东西和提交 574f4c3 完全一样。** 不改 `Params` 和 `Settings` 的默认值，不改上传、重建、发布质量门、结构、风格化和 HTTP 接口做的事。每一步只要动了 `crowd3d/` 或 `scripts/` 下的代码，就要拿出这几样证明：

- 重建的金样 `tests/fixtures/recon_golden/golden.json` 逐字节不变；
- 各个版本号不变，并说明为什么不用变；
- 目录里 13 个地标的素模 GLB 改之前和改之后逐字节相同；
- `eval/results/LEADERBOARD.md` 重新生成后和仓库里的文件相同。

**关于数字**：都是在开发机（4 核，Intel Xeon 2.3 GHz，无 GPU，Python 3.13.16，Node 22.22.0）上实测的，测的时候机器上没有别的 COLMAP 或测试在跑。"改之前"指提交 574f4c3。

---

## 第 1 步：依赖文件、pytest 配置、并行测试、代码检查、CI

对应评审"工程做法"的这几条：

- 没有 CI，也没有提交的代码检查配置；
- 快速测试慢：不能并行，也没有按领域分的标记；
- `requirements.txt` 不完整，也不是锁文件：缺 pyarrow；starlette、pydantic、anyio 被直接导入却没钉版本；
- 直接运行 `pytest` 会失败（核查者补的一条）；
- "少量死代码和 lint 残留"里的 lint 部分：没用到的导入和变量、多余的 `noqa`（没人引用的定义留给下一步）；
- "测试文件互相导入"那一条里，并行运行用得上的部分。

另外改了两个已知偶尔会失败的测试。

### 1. 依赖文件

**哪里不对**：

- 只有一个 `requirements.txt`，10 个包。`scripts/` 和测试要用的 pyarrow 不在里面：干净环境照 README 安装后，`tests/test_materialize_testset.py` 有 9 个测试报错。
- `crowd3d/` 直接导入了 starlette、pydantic 和 anyio，清单里没有，版本由 FastAPI 带进来的是哪个算哪个。
- 运行服务用不到的 pytest 和 httpx 和运行依赖混在一起。
- 没有锁文件：间接依赖换了版本，谁也看不出来。

**改了什么**：拆成三层，每一层用 `-r` 包含下面一层，每个包都写成 `名字==版本`，版本就是仓库 `.venv` 里装着的那个。

| 文件 | 给谁用 | 自己列的包 |
|---|---|---|
| `requirements.txt` | 只运行服务 | numpy、opencv-python-headless、pycolmap、fastapi、starlette、pydantic、anyio、uvicorn、python-multipart、piexif、pillow |
| `requirements-eval.txt` | 再加上 `scripts/` 和评测 | httpx、pyarrow |
| `requirements-dev.txt` | 再加上测试和检查（开发、CI 装这个） | pytest、pytest-xdist、ruff |
| `requirements-lock.txt` | 约束文件（`pip install … -c requirements-lock.txt`） | 干净环境的 `pip freeze`，32 个包：上面 16 个，加上它们带进来的 16 个 |

新增的版本：starlette 1.9.0、pydantic 2.14.1、anyio 4.15.1（原来就装着，只是没写）、pyarrow 26.0.0（同上）、pytest-xdist 3.8.0、ruff 0.17.4（这两个是这一步新装进 `.venv` 的，pytest-xdist 带进来 execnet 2.1.2）。原有 10 个包的版本一个没动。

**导入是怎么找的**：`tests/test_tooling.py` 用 Python 的 `ast` 把三个目录的每个 `.py` 文件走一遍：`import` 和 `from … import`（函数里面的也算），以及参数是字符串的 `__import__`、`import_module`、`importorskip`；去掉标准库和仓库自己的模块，再用 `importlib.metadata.packages_distributions()` 把模块名换成包名（`PIL` → pillow，`cv2` → opencv-python-headless）。扫出来的结果：

| 目录 | 直接导入的第三方包 |
|---|---|
| `crowd3d/` | pillow、anyio、opencv-python-headless、fastapi、numpy、piexif、pycolmap、pydantic、starlette、uvicorn |
| `scripts/` | pillow、opencv-python-headless、fastapi、httpx、numpy、pyarrow、pycolmap |
| `tests/` | pillow、anyio、opencv-python-headless、fastapi、numpy、piexif、pyarrow、pycolmap、pytest、uvicorn |

有三个包没有任何文件导入，但是要装，测试里各写了一句理由：python-multipart（FastAPI 读上传表单要用，没有它 `create_app` 直接抛 `RuntimeError`，实测过）、pytest-xdist（`pytest -n`）、ruff。

**不让它再走样**：`tests/test_tooling.py` 里 23 个快速测试，其中管依赖的 22 个：

- 每个目录导入的第三方包都钉在它那一层的文件里（或者那个文件包含的文件里）；
- 反过来，文件里列的每个包，它那一层确实有文件导入（上面三个例外除外）；
- 每一行都是 `名字==版本`，同一个包不在两层里各钉一次；
- 钉的 16 个包，装着的版本就是钉的版本（一个包一个测试；没装的包是跳过，不是失败，因为不少测试本来就允许没有 pycolmap）；
- 锁文件里这 16 个包的版本和三个文件一致，并且每一行都是 `名字==版本`。

拿 9 种改坏的写法试过，每一种都有测试失败：从 `requirements-eval.txt` 删掉 pyarrow、从 `requirements.txt` 删掉 starlette、把 anyio 从服务那一层挪到开发那一层、往 `requirements.txt` 加一个没人导入的 click、把 numpy 钉成另一个版本、把 `==` 写成 `>=`、锁文件里 fastapi 的版本不一样、锁文件里没有 ruff、往 `scripts/` 加一个导入 yaml 的文件。

**锁文件是怎么来的**：在一个全新的虚拟环境里 `pip install -r requirements-dev.txt`，用仓库 `.venv` 的 `pip freeze` 当约束，装完的 `pip freeze` 和仓库 `.venv` 的逐行相同（32 行）。也就是说，三套测试一直在跑的那个环境，现在原样写进了 `requirements-lock.txt`。锁文件没有哈希（见"这一步没做的"）。

**在干净环境里验证**：一个全新的虚拟环境（`python3.13 -m venv`），照 README 写的那一行安装（`pip install -r requirements-dev.txt -c requirements-lock.txt`；包已经在本机的 pip 缓存里时约 10 秒）。装完的 `pip freeze` 和锁文件逐行相同（32 行）。在这个环境里快速测试 1,420 个全部通过、慢测试 40 个全部通过，数量和仓库的 `.venv` 里一样。包索引是通过本机配置的代理访问的，TLS 校验没有关。验证完这个环境就删掉了。

### 2. pytest 配置

**哪里不对**：

- 在仓库根目录直接运行 `pytest`，第一步就是 `ModuleNotFoundError: No module named 'crowd3d'`：包没有安装，`pyproject.toml` 也没有 `pythonpath`，只有 `python -m pytest` 能用（它会把当前目录放进 `sys.path`）。
- 标记只登记了 `slow` 一个，也没开 `--strict-markers`：标记写错了不报错，只是一个测试都选不中。
- 不能只跑一个领域。改一行 `api.py`，也要等结构那一族跑完。

**改了什么**（`pyproject.toml` 的 `[tool.pytest.ini_options]`）：

- `pythonpath = ["."]`。`pytest` 和 `python -m pytest` 现在收集到同样的 1,460 个测试（两份 id 列表逐行相同）；在 574f4c3 的导出里直接运行 `pytest`，报的就是评审写的那个 `ModuleNotFoundError`。
- `addopts = ["--strict-markers"]`。打开以后没有发现写错的标记：现有的标记只有 `slow` 和 pytest 自带的几个。
- 登记了两个领域标记，按模块加（每个文件一行 `pytestmark`）：

| 标记 | 含义 | 文件 |
|---|---|---|
| `structure` | 素模那一族：结构、风格化、浮雕、裁决、D15 / D24 / D25 | 18 个：`test_stylize`、`test_structure`、`test_structure_pipeline`、`test_structure_properties`、`test_structure_stylize`、`test_relief`、`test_d15`、`test_d24`、`test_d25`、`test_d25_measure`、`test_d25_publish`、`test_d29_pins`、`test_lines3d`、`test_skyline_judge`、`test_snap_gate73`、`test_structmetrics`、`test_catalogue_structure`、`test_e2e_stylize` |
| `harness` | `scripts/` 的测试：基准、评测工具、Commons 和测试集工具 | 7 个：`test_benchmark`、`test_eval_real`、`test_fetch_commons`、`test_materialize_testset`、`test_export_for_splatting`、`test_replay_map_views`、`test_trim_osm` |
| （都不带） | 服务和重建：上传、校验、存储、作业、API、地理配准、定位、网页，以及这一步加的 `test_tooling` | 其余 23 个文件 |

用法：`-m "not slow and not structure"`（去掉素模那一族）、`-m structure`、`-m harness`、`-m "not slow and not structure and not harness"`（只跑服务和重建）。

**每一族要多久**（代码改完以后单进程跑 3 次，每个文件取中位数；时间是 pytest 记的每个测试的准备 + 运行 + 收尾。只有慢测试的 `test_e2e` 和 `test_e2e_stylize` 不在表里）：

| 一族 | 文件 | 测试 | 秒 | 占比 |
|---|---|---|---|---|
| `structure`（素模） | 17 | 499 | 201.4 | 66.7% |
| 不带标记（服务和重建） | 22 | 686 | 86.9 | 28.8% |
| `harness`（`scripts/`） | 7 | 235 | 13.7 | 4.5% |
| 合计 | 46 | 1420 | 302.1 | 100% |

改之前（574f4c3，1,396 个测试，同样的测法跑 1 次）是 203.4 s、88.5 s、15.2 s，合计 307.1 s。评审按单价外推，估计素模那一族占 78%；实测是三分之二。

去掉素模那一族（`-m "not slow and not structure"`，921 个测试）：单进程 101.2 s，`-n 4` 43.0 s。只跑服务和重建（686 个测试）：`-n 4` 38.4 s。

**每个文件要多久**（同一组数据，从慢到快）：

| 文件 | 一族 | 测试 | 秒 |
|---|---|---|---|
| `test_d25_measure` | structure | 22 | 38.3 |
| `test_relief` | structure | 81 | 32.5 |
| `test_stylize` | structure | 71 | 30.5 |
| `test_georegister` | — | 53 | 26.2 |
| `test_api` | — | 130 | 22.9 |
| `test_snap_gate73` | structure | 7 | 18.0 |
| `test_d15` | structure | 34 | 16.5 |
| `test_d24` | structure | 21 | 15.9 |
| `test_structure_properties` | structure | 40 | 11.2 |
| `test_geo_exif` | — | 63 | 10.5 |
| `test_structure_pipeline` | structure | 15 | 10.2 |
| `test_lines3d` | structure | 20 | 9.5 |
| `test_eval_real` | harness | 80 | 8.2 |
| `test_ingest` | — | 57 | 6.8 |
| `test_jobs` | — | 50 | 4.4 |
| `test_catalogue_structure` | structure | 6 | 4.3 |
| `test_gzipfiles` | — | 53 | 3.9 |
| `test_structure_stylize` | structure | 19 | 3.7 |
| `test_d29_pins` | structure | 11 | 3.7 |
| `test_synthetic` | — | 10 | 3.6 |
| `test_skyline_judge` | structure | 14 | 2.7 |
| `test_structure` | structure | 58 | 2.4 |
| `test_materialize_testset` | harness | 26 | 2.2 |
| `test_tooling` | — | 23 | 1.8 |
| `test_benchmark` | harness | 78 | 1.5 |
| `test_fetch_commons` | harness | 39 | 1.5 |
| `test_web` | — | 1 | 1.3 |
| `test_d25_publish` | structure | 13 | 1.1 |
| `test_validation` | — | 46 | 0.9 |
| `test_recon_golden` | — | 2 | 0.9 |
| `test_d25` | structure | 48 | 0.8 |
| `test_subject` | — | 22 | 0.7 |
| `test_store` | — | 21 | 0.7 |
| `test_reconstruct` | — | 3 | 0.7 |
| `test_photoposes` | — | 23 | 0.6 |
| `test_trim_osm` | harness | 5 | 0.3 |
| `test_contributors` | — | 28 | 0.3 |
| `test_csp` | — | 11 | 0.3 |
| `test_localize` | — | 39 | 0.2 |
| `test_landmark_map` | — | 8 | 0.1 |
| `test_structmetrics` | structure | 19 | 0.1 |
| `test_anchor` | — | 8 | 0.0 |
| `test_export_for_splatting` | harness | 4 | 0.0 |
| `test_qa` | — | 19 | 0.0 |
| `test_coverage` | — | 16 | 0.0 |
| `test_replay_map_views` | harness | 3 | 0.0 |

**测试集合没有变**：改之前收集到 1,436 个测试 id（快速 1,396，慢 40），改之后 1,460 个（快速 1,420，慢 40）。改之前的 1,436 个 id 改之后一个不少、名字不变；多出来的 24 个都是这一步新加的：`tests/test_tooling.py` 23 个，`tests/test_api.py` 1 个（`test_a_server_started_inside_a_test_leaves_the_allocator_of_the_test_process_alone`）。

测试本身改了多少，用脚本比过：把每个测试文件里每个顶层定义（函数、类）的语法树和改之前逐个对比。除了 25 个模块各加的一行 `pytestmark` 和几处导入的增删，改过的定义只有 5 个：`test_api.py` 的 404 列表那个测试和启动真实服务的 `real_server`，`test_geo_exif.py` 的计时函数 `cost_of`，`test_gzipfiles.py` 里用真实服务的那个测试（只改了启动方式），`test_jobs.py` 一个测试里去掉了一个没用到的变量名。前四个的理由见下面"并行运行"一节。

### 3. 并行运行

**哪里不对**：`.venv` 里没有 pytest-xdist，4 个核只用 1 个。

**改了什么**：pytest-xdist 加进 `requirements-dev.txt`；找出并行时会互相影响的测试，改掉。推荐的写法是 `python -m pytest -q -m "not slow" -n 4`。

#### 找到的三处共享状态

先拿改之前的测试直接用 `-n 4` 跑，再拿一个记录"哪个测试之后进程的内存分配器被钉住"的临时插件跑了一遍，又在 8 个空转进程占着 4 个核的情况下把带时间断言的几个文件跑了一遍。找到三处，都改了：

**(a) 测试进程的内存分配器被别的测试钉住。** 服务启动时会给自己所在的进程钉住 glibc 的 mmap 阈值（`ingest.pin_malloc_threshold`，D34 第 1 步），钉了就改不回去。`conftest.py` 的 `settings` 夹具把这个开关关了，但自己造 `Settings` 的代码没有关：快速测试里是评测工具（`tests/test_benchmark.py` 2 个测试、`tests/test_eval_real.py` 6 个测试通过 `run_crowd` / `rescore` 在测试进程里用默认设置启动服务），慢测试里是端到端的几个模块级夹具（`test_e2e.py`、`test_e2e_stylize.py`、`test_localize.py`）。同一个进程里排在它们后面的所有测试都在钉住的分配器上跑；哪些测试受影响取决于运行顺序，并行时还取决于分到哪个进程。

改法：`conftest.py` 加两个对所有测试自动生效的夹具。

- `server_starts_leave_the_allocator_alone`（整个会话一个）：测试进程里，服务启动时的那次调用什么都不做。它必须是会话级的：模块级夹具启动服务的时候，测试自己的夹具还没建起来（第一版写成了每个测试一个，读慢测试的夹具时发现挡不住它们，改了）。
- `the_test_process_keeps_its_allocator`（每个测试一个）：测试结束时如果进程被钉住了，这个测试自己失败，而不是让后面的测试换一个环境跑。

钉住分配器这件事本身照旧有测试：`tests/test_ingest.py` 用一个记录调用的假 C 库测逻辑，`tests/test_api.py` 的三个真实服务进程测效果（那是子进程，不受夹具影响）。新加的测试 `test_a_server_started_inside_a_test_leaves_the_allocator_of_the_test_process_alone` 守着第一个夹具：去掉它，这个测试就失败。

**(b) 真实服务进程的端口。** `tests/test_api.py` 的 3 个测试和 `tests/test_gzipfiles.py` 的 1 个测试各启动一个真实的服务进程。原来的做法是先向内核要一个空闲端口、关掉、再把端口号交给服务；关掉和服务绑定之间有空当，两个测试进程同时这么做时可能拿到同一个端口，之后一个测试会和另一个测试的服务说话。概率很小（两次取端口要正好撞上同一个号），但并行以后是真的可能。

改法：新文件 `tests/server_process.py`，服务用 `--port 0` 启动，由内核在绑定时分配端口，测试从 uvicorn 的启动日志里读出端口号。两个文件原来各有一份几乎相同的启动代码，现在共用这一份。服务端代码没有改。

**(c) 一个按墙上时钟计时的断言余量太小。** `tests/test_geo_exif.py::test_a_header_flood_is_unreadable_before_pillow_sees_it` 断言"拒绝一个 24 MB 的坏文件头用时不到 0.5 秒"，计时用的是墙上时钟。最慢的三种文件在空闲的机器上就要 0.2–0.36 秒，机器一忙就超：第一次 `-n 4` 就失败了 1 个，8 个空转进程时失败 3 个。

改法：这个文件里量"一次调用花多少"的函数 `cost_of` 改成量本进程的 CPU 时间（`time.process_time`），0.5 秒的界线不变。被量的调用只解析内存里的字节，不等任何东西，所以机器空闲时两种时间是同一个数；机器忙时墙上时钟把别的进程也算进去了：

| 最慢的一种（2,000,000 个空的 PNG 数据块，24 MB） | 墙上时钟 | 本进程 CPU 时间 |
|---|---|---|
| 机器空闲，5 次 | 0.287–0.295 s | 0.287–0.295 s |
| 8 个空转进程占着 4 个核，5 次 | 1.265–1.491 s | 0.305–0.321 s |

这样改以后，这个断言不再能发现"调用在等待而不是在计算"这一类变慢；被量的函数（`read_photo_meta`、`_read_exif`）不做读写、不拿锁，没有可等的东西。

#### 查过、不用改的

- 测试写文件都在 `tmp_path` 里；整套快速测试跑完，仓库目录里除了缓存目录（`.pytest_cache`、`__pycache__`）没有新文件。
- 环境变量、`PIL.Image.MAX_IMAGE_PIXELS`、模块里的常量都是用 `monkeypatch` 改的，测试结束自动还原。
- 模块级的缓存（`structure_pipeline._LOADED`、`subject._memory`、`jobs` 和 `synthetic` 的 `lru_cache`、`params` 的当前参数）每个进程一份，并行时互不相干；同一个进程里换顺序是否有影响，由下面的"换顺序"检查。
- 会话级夹具 `synthetic_dataset`（生成合成照片，约 3 秒）每个进程各建一次。
- `tests/test_e2e.py` 的真实 uvicorn 用的就是端口 0（它是慢测试，不并行）。
- 其余按墙上时钟计时的断言（`test_geo_exif.py` 3 处、`test_ingest.py` 3 处、`test_synthetic.py` 1 处、`test_jobs.py` 2 处）在 8 个空转进程占着 4 个核时全部通过，余量够，没有动（例如合成照片的渲染界线是 2.5 秒，本机每张 0.14 秒）。它们在更慢的机器上会不会超，见 CI 一节。

没有用到 `xdist_group` 和 `--dist loadgroup`：三处都改成了各用各的，没有哪个测试必须留在同一个进程里。

#### 两个已知偶尔失败的测试

**(i) `tests/test_api.py` 里"不是这个地标的行就一律 404"的列表。** 列表里有 `row_id.upper()`。公开 id 是 12 位十六进制，恰好不含字母时（(10/16)¹² ≈ 0.36%，约 280 次里 1 次）大写和原来是同一个 id，于是测试拿一个有效的 id 去期待 404。

- 复现：一个临时插件把每个公开 id 里的 a–f 换成 0–5，这个测试必然失败（200 而不是 404）。
- 改法：列表里放的是这个地标所有带字母的行的大写形式；一行都不带字母时这一项为空（那时本来就没有"只差大小写"的 id）。原来只试第一行，现在每一行都试。
- 验证：同一个插件下通过；不带插件通过；把照片接口改成不分大小写（变异），测试失败。

`tests/test_api.py` 里另有 4 处把凭证转成大写再发出去，对象是 64 位十六进制的凭证，不含字母的机会是 (10/16)⁶⁴ ≈ 10⁻¹³，没有动。

**(ii) `test_the_server_pins_the_malloc_threshold_when_it_starts`。** 它最后断言"没有哪个测试钉住过这个进程"。按文件名顺序跑时 `test_api.py` 排在 `test_benchmark.py` 前面，所以一直通过；先跑 `test_benchmark.py` 或 `test_eval_real.py` 就失败。

- 复现：`pytest tests/test_benchmark.py tests/test_api.py::test_the_server_pins_the_malloc_threshold_when_it_starts`，1 个失败（`assert 4194304 is None`）。
- 改法：就是上面的 (a)。这个测试本身一个字没改，它断言的那句话现在是真的。
- 验证：`test_benchmark.py`、`test_eval_real.py`、这个测试、`test_ingest.py` 按这个顺序一起跑，216 个全部通过。

#### 换顺序

整套快速测试按三种别的顺序各跑了一遍，都是单进程：

| 顺序 | 结果 | 用时 |
|---|---|---|
| 测试文件按名字倒着排（文件里的顺序不变） | 1,420 个通过 | 302.5 s |
| 整个列表倒过来（文件倒着排，文件里的测试也倒着排） | 1,420 个通过 | 298.2 s |
| 文件的顺序和每个文件里测试的顺序都打乱（固定的随机种子 20261112） | 1,420 个通过 | 303.0 s |

改之前（574f4c3 的导出），文件倒着排时因为顺序而失败的只有上面的 (ii)（那次运行里还有 1 个失败和顺序无关：`test_code_fingerprint_covers_the_code_a_run_executes` 要求代码树是 git 检出，单独跑它在导出的目录里也失败）。改之后没有发现别的顺序依赖。后两种顺序用的是一个只为这次检查写的临时插件，没有进仓库。

#### 实测：单进程和 4 个进程

用时是 pytest 自己报的秒数。跑的时候机器上没有 COLMAP，也没有别的测试。

| 写法 | 结果 | 用时 |
|---|---|---|
| 单进程（基准），3 次 | 每次 1,420 个通过 | 303.6 s、302.9 s、303.7 s（约 5 分 4 秒） |
| `-n 4`，3 次 | 每次 1,420 个通过 | 88.5 s、88.3 s、89.8 s（约 1 分 29 秒） |
| `-n 2` | 1,420 个通过 | 158.2 s |
| `-n auto`（本机就是 4 个进程） | 1,420 个通过 | 88.0 s |
| `-n 8`（进程数是核数的两倍，看余量） | 1,420 个通过 | 93.1 s |
| `-n 4 --dist loadfile`（一个文件的测试留在一个进程里） | 1,420 个通过 | 96.1 s |
| `-n 4 --dist worksteal` | 1,420 个通过 | 87.7 s |

4 个进程比单进程快 3.4 倍（中位数 303.6 s 对 88.5 s）。到不了 4 倍，是因为每个进程都要各自导入代码、各建一次合成照片和用到的模块级场景。默认的分发方式和 `worksteal` 差不多；`loadfile` 慢一些（最大的几个文件各占住一个进程）。所以推荐的写法不带 `--dist`。

上表里的 8 次并行运行全部通过；改之前第一次 `-n 4` 就有 1 个失败（上面的 (c)）。

#### `-n` 不写进 `addopts`

不写。理由：

- 单进程的写法是基准，必须照旧能用，而且不依赖 pytest-xdist 装没装（写进 `addopts` 以后，没装 xdist 的环境里 `pytest` 直接报"不认识的参数"）。
- 慢测试跑真实的 COLMAP，必须一个一个跑；写进 `addopts` 就得每次记着加 `-n 0`。
- 调试时要 `-s`、`--pdb`，这两个在并行时用不了。

所以并行是明写的：README 的命令和 CI 里各写一次。

### 4. 代码检查

**哪里不对**：仓库里没有任何检查配置，代码里却留着给 ruff 看的 `noqa` 注释；没人在看，所以有 6 条 pyflakes 的发现和几十处多余的 `noqa` 留着。

**改了什么**：`pyproject.toml` 加了 `[tool.ruff]`，ruff 钉在 0.17.4。

- 目标 Python 版本取自 `requires-python`（3.11），行宽 120（代码现在就是按 120 折行的）。
- 规则：pyflakes（`F`）、pycodestyle 的错误和警告（`E`、`W`）、多余的 `noqa`（`RUF100`），再加上七组仓库本来就一条不违反的规则：pylint 的错误类（`PLE`）、NumPy 的过时用法（`NPY`）、异步函数里的阻塞调用（`ASYNC`）、遗留的调试器（`T10`）、logging 的误用（`LOG`）、不带规则号的 `noqa`（`PGH`）、FastAPI 的路由声明（`FAST`）。
- 两条不查：`E501`（行太长：74,266 行里有 626 行超过 120，占 0.8%，重新折行是格式化工具的事）和 `E731`（把 lambda 赋给名字：代码里有 26 处，是有意这么写的短辅助函数）。
- ruff 0.17 不带配置时的默认规则集比这个大得多（改之前在本仓库报 377 条，多数是写法上的建议），所以规则是明写的。

**没有跑格式化工具。** `ruff format --check` 说 115 个文件里有 99 个要重排，一次改几千行，和这一步"产品算的东西不变"的证明混在一起就没法看了。行宽已经写在配置里，以后要格式化时单独做一个只含格式化的提交。

**一条一条改掉的**（改完 `ruff check .` 报 0 条）：

| 发现 | 位置 | 怎么改的 |
|---|---|---|
| `F541` 没有占位符的 f 字符串 | `crowd3d/judge.py` | 去掉 `f` 前缀（编译出的字节码相同） |
| `E702` 一行里用分号写了三个语句，4 行 | `crowd3d/roof_fit.py` | 拆成 12 行 |
| `E741` 变量名 `O` | `crowd3d/lines3d.py` | 没有改名，加了带理由的 `noqa`（改名会让"语法树相同"的证明少一个文件） |
| `E402` 导入不在文件开头，5 行 | `scripts/stylize_model.py` | 这 5 行导入必须排在 `REPO = …` 和 `sys.path.insert` 后面，加了带理由的 `noqa` |
| `E401` 一行导入两个模块 | `tests/metrics_glb.py` | 拆成两行 |
| `F401` 没用到的导入，2 处 | `tests/test_d25_publish.py`（`structmetrics`）、`tests/test_jobs.py`（`ReconstructionResult`） | 删掉。先查过：没有别的文件从这两个测试文件导入它们，`structmetrics` 导入时也不做任何登记 |
| `F841` 赋了值没用的变量 | `tests/test_jobs.py` | 去掉赋值，调用留着 |
| `F811` 重复定义，2 处 | `tests/test_materialize_testset.py` | 这是 pytest 按名字导入别的文件的夹具的写法，加了带理由的 `noqa`；把夹具挪进 `conftest.py` 留给整理测试的那一步 |
| `RUF100` 多余的 `noqa: E402`，40 处 | `scripts/` 下 7 个文件 | 删掉注释。ruff 本来就允许 `sys.path.insert` 之后再导入 |
| `RUF100` 给没启用的规则写的 `noqa`，15 处 | `BLE001` 5 处、`E731` 8 处、`S310` 1 处、文件级的 `RUF046` 1 处 | 去掉 `noqa` 这个词，后面写的理由留下来当普通注释 |

`crowd3d/stylize.py` 给别的模块转出的名字（`st.名字`）和为了副作用做的导入（`crowd3d/exif.py` 里登记图片格式的那几个 PIL 插件、`crowd3d/localize.py` 里试探 pycolmap 装没装的那一句）都带着 `noqa: F401`，没有动。改的过程中出过一次错：用 `ruff check --select RUF100 --fix` 自动删多余的 `noqa` 时，它把所有别的规则的 `noqa` 都当成"没启用"删掉了（15 处，包括上面这几个）；从 git 恢复了这些文件，之后只用仓库的配置跑。最后的差异逐行看过：删掉的 `noqa` 正好是上表的 55 处，一处不多。

**怎么保持**：两处都查。快速测试里的 `test_ruff_has_nothing_to_report` 用当前环境的 ruff 跑 `ruff check .`（没装 ruff 时跳过并说明原因）；CI 里另有单独的一步跑同一条命令。

**类型检查不在这一步**，见"这一步没做的"。

### 5. CI

新文件 `.github/workflows/ci.yml`（单独一个提交，是这一步的最后一个提交）。

| 任务 | 什么时候跑 | 做什么 |
|---|---|---|
| `fast` | 每次推送、每个拉取请求 | Python 3.13，`pip install -r requirements-dev.txt -c requirements-lock.txt`，`ruff check .`，快速测试（`-n auto`，有几个核开几个进程），Node 22 跑 `node --test 'web/tests/*.test.mjs'` |
| `slow` | 手动触发，每周一 03:17 UTC | 同样的安装，`python -m pytest -q -m slow`，单进程 |

**慢测试为什么不是每次推送都跑**：它跑真实的 COLMAP（通过 pycolmap 的 wheel），再把目录里 13 个地标各建一遍，在 4 核的开发机上约 8.5 分钟；它查的是重建和作业，多数推送不碰这些。每周跑一次，再加上改到这些代码时手动触发，比让每次推送都等它划算。注意 GitHub 只从默认分支运行定时和手动触发的工作流：这个文件合进默认分支之前，`slow` 任务跑不起来。

**其他设定**：

- 权限最小：`permissions: contents: read`，检出代码时不保留凭证（`persist-credentials: false`）；不用任何 secret；不上传任何产物（测试只用合成照片）。
- `concurrency`：同一个分支（或拉取请求）、同一种事件的新一次运行会取消还在跑的旧的。
- pip 的下载缓存按四个依赖文件的内容做键。
- 超时：`fast` 30 分钟，`slow` 90 分钟。
- 用到的 action 钉在主版本：`actions/checkout@v5`、`actions/setup-python@v6`、`actions/setup-node@v5`；运行环境钉在 `ubuntu-24.04`。
- 快速测试在 CI 里用 `-n auto` 而不是 `-n 4`：GitHub 的机器是 2 核还是 4 核取决于仓库是私有还是公开。

**需要 runner 上没有的东西的测试**：读过全部测试，没有哪个测试需要浏览器、字体或 GPU。

- 浏览器：真实浏览器里的核对（Playwright）一直是手工做的，不在三套测试里。
- 字体：合成照片上的字用的是 OpenCV 自带的 Hershey 字体，不读系统字体。
- GPU：pycolmap 用 CPU 提特征。
- 系统库：opencv-python-headless 和 pycolmap 的 wheel 自带依赖的库，对系统只要求 glibc、libstdc++、libz（用 `ldd` 看过），Ubuntu 24.04 都有，不用 `apt install`。
- 有一个快速测试要求代码树是 git 检出、机器上有 `git`（`test_code_fingerprint_covers_the_code_a_run_executes`，它用 `git ls-files` 列出代码文件）。CI 的 `actions/checkout` 给的就是 git 检出，浅克隆也够用；从压缩包解出来的代码树上它会失败。
- 有条件跳过的测试只有这些：找不到 `node` 时跳过 `tests/test_web.py`（CI 装了 Node 22）；不是 Linux + glibc 时跳过两个读 `/proc` 的内存测试（CI 是）；没装 pycolmap 时跳过用到它的测试（CI 装了）；node 测试里用 Khronos glTF 校验器的那一个要设两个环境变量才跑（CI 不设，一直是跳过）。

**CI 不覆盖的**：

- 基准（`scripts/benchmark.py run`）：要仓库外的真实照片。
- 真实浏览器：页面在 Chromium 里的样子、内容安全策略在浏览器里是否放行。
- glTF 校验器那一个 node 测试。
- Python 3.11 和 3.12：`requires-python` 写的是 ≥ 3.11，CI 只跑 3.13（钉的版本就是在 3.13 上装出来的）。
- macOS 和 Windows。

**在本机验证过的**：

- 文件能被 YAML 解析；`actionlint` 1.7.12 检查通过，0 条（它是从 PyPI 的 `actionlint-py` 装进一个临时环境的，没有进依赖文件）。
- 在上面那个干净环境里，把工作流文件里的命令原样跑了一遍。`fast` 任务：安装、`python -m ruff check .`、`python -m pytest -q -m "not slow" -n auto --durations=25`（1,420 个通过，约 1 分 31 秒）、`node --test 'web/tests/*.test.mjs'`（211 个：210 个通过，1 个跳过）。`slow` 任务：安装、`python -m pytest -q -m slow --durations=25`（40 个通过）。
- 本机验证不了的是 GitHub 那一侧：触发条件、action 能不能下载、缓存、并发取消。

**第一次在 GitHub 上跑的时候要看的**（这一步看不到 GitHub，下面这些只能在第一次运行里确认）：

- 三个 action 的主版本号能不能解析。写的是已知存在的主版本；如果 GitHub 报找不到，改成能用的主版本即可。
- 推送带工作流文件的提交需要 `workflow` 权限，可能被拒；所以这个文件单独放在最后一个提交里，前面的提交可以先推。
- 时间断言。几处按时间设的界线是在这台开发机上量的，runner 的 CPU 慢的话可能超：`test_geo_exif.py` 的坏文件头（CPU 时间 < 0.5 s，本机最慢 0.31 s，只有 1.6 倍余量，最可能超的就是它）、别的几处余量大得多（`test_synthetic.py` 的渲染界线 2.5 s，本机 0.14 s；`test_ingest.py` 的三处界线 1–2 s）。超了不要放宽界线了事，先看 `--durations=25` 的输出和失败信息里的实测值。
- 核数。2 核的 runner 上 `-n auto` 只开 2 个进程，快速测试估计要 3–4 分钟；慢测试里 COLMAP 是多线程的，2 核上会比本机慢不少（超时给了 90 分钟）。
- 真实服务进程的 4 个测试要在本机回环地址上开端口、读 `/proc/<pid>/status`，GitHub 的 Ubuntu runner 上应该都可以。
- pip 缓存第一次是空的，安装约 600 MB 的包要一两分钟。

### 证明：产品算的东西没有变

这一步动了 `crowd3d/` 下 7 个文件、`scripts/` 下 9 个文件，全部是注释和排版，外加一个 f 前缀：

- **程序相同**：把改之前（574f4c3 的导出）和改之后 `crowd3d/`、`scripts/` 下的每个文件用 `ast` 解析后比较（不比行号和列号）：15 个改过的文件语法树完全相同；剩下的 `crowd3d/judge.py`（去掉 f 前缀）语法树不同，编译出的字节码相同（不比行号表）。没有一个文件是"不同的程序"。
- **金样**：`tests/fixtures/recon_golden/golden.json` 没有改动（`git diff` 为空），快速测试里的金样测试通过。
- **13 个地标的 GLB**：从 574f4c3 的导出和改之后的代码各建一遍目录里 13 个地标的素模（不带照片，也就是产品现在给出的 OSM 分件素模）。13 个 GLB 逐字节相同，每个分件的记录也相同，一共 824 个分件。
- **排行榜**：`python -I scripts/benchmark.py report --out <临时文件>` 生成的文件和仓库里的 `eval/results/LEADERBOARD.md` 逐字节相同（38 次运行，sha256 都是 `7a92e8c8…`）。

### 测试

都在提交的那棵代码树上跑：先 `git add`，记下 `git write-tree`，跑完核对没有变再提交。

| 套件 | 在哪个环境 | 结果 |
|---|---|---|
| 快速测试，单进程 | `.venv` | 1,420 个通过（约 5 分钟） |
| 快速测试，`-n 4` | `.venv` | 1,420 个通过（约 1.5 分钟） |
| node 测试 | Node 22.22.0 | 211 个：210 个通过，1 个跳过（要 glTF 校验器的那个，一直是跳过） |
| 快速测试，单进程 | 干净环境 | 1,420 个通过 |
| 慢测试（CI 里的那条命令） | 干净环境 | 40 个通过（约 8.5 分钟） |

带工作流文件的那个提交只多一个 `.github/workflows/ci.yml`，没有测试读它。在那棵树上又跑了 `fast` 任务的全部命令（干净环境）和单进程的快速测试（`.venv`），没有再跑慢测试。

和改之前比：快速测试 1,396 → 1,420，多的 24 个见"pytest 配置"一节；慢测试 40 个、node 测试 211 个，没有变。没有哪个测试被跳过、标成预期失败、删掉或者放宽；断言改过的三处（404 的列表、`cost_of` 的计时、真实服务的启动方式）各自的理由见"并行运行"一节。

### 版本号

都没有变，因为 `crowd3d/` 和 `scripts/` 编译出来的程序没有变（见上一节）：

| 版本号 | 值 | 为什么不用变 |
|---|---|---|
| `RECON_OUTPUT_VERSION`（`crowd3d/reconstruct.py`） | 4 | 这个文件一个字节没动 |
| `POSES_VERSION`、`PUBLIC_PHOTO_VERSION`（`crowd3d/photoposes.py`） | 2、1 | 这个文件一个字节没动 |
| `HARNESS_VERSION`、`METRIC_SCHEMA_VERSION`、`REFERENCE_VERSION`（`scripts/eval_real_landmark.py`） | 8、8、1 | 这个文件只删了 11 处 `noqa: E402` 注释，语法树相同 |
| `STYLE_VERSION`（`crowd3d/stylize.py`） | 2 | 这个文件只改了一行注释，语法树相同；13 个 GLB 逐字节相同 |
| `MASK_VERSION`（`crowd3d/structure_pipeline.py`） | `{"A+R": 1}` | 这个文件只改了一行注释，语法树相同 |
| `LOC_VERSION`（`localize.py`）、`SUBJECT_VERSION`（`subject.py`）、`RESULT_SCHEMA_VERSION`（`benchmark.py`）、`SCHEMA_VERSION`（`materialize_testset.py`） | 都是 1 | 前两个文件没动；后两个只动了注释 |

`crowd3d/structure.py` 自己没有版本号。用脚本把 `crowd3d/` 和 `scripts/` 里名字带 `VERSION` 的模块级常量（18 个）逐个比过，改之前和改之后的写法全部相同。

### 和任务说明不一样的地方

- **httpx 放在 `requirements-eval.txt`，不在 `requirements-dev.txt`。** 任务说明把它列在开发那一层。但 `scripts/demo.py` 直接导入它，评测工具用的 `TestClient` 也要它，按"哪一层导入就钉在哪一层"的规则它属于评测那一层；开发那一层通过 `-r` 一样装到它。同一个包钉两处，测试会拒绝。
- **CI 里快速测试用 `-n auto`，不是 `-n 4`。** 理由见 CI 一节（runner 可能只有 2 核）。本机上 `-n auto` 就是 4 个进程。
- **没有用 `xdist_group` / `--dist loadgroup`。** 任务说明给了两条路：改成各用各的，或者留在一个进程里。三处都走了第一条。
- **`E741` 用 `noqa` 而不是改名**；`E731` 整条不查，而不是把 26 处 lambda 改成 `def`。两处都是不想因为检查工具去改代码的写法。
- **README 除了安装和测试命令，还改了测试一节里的数量和"最近一次完整运行"那一条**：这两处的数字因为这一步变了，不改就是错的。另外在 README 的文档列表和 [D34-review-fixes.md](D34-review-fixes.md) 的开头各加了一句指向本文的话。

### 这一步没做的

- **类型检查**（mypy 或 pyright）：没装，没配。代码里类型标注不少，但从没检查过，第一次开会有一批要处理，值得单独一步。
- **格式化**：没有跑 `ruff format`，理由见上。
- **锁文件的哈希**：`requirements-lock.txt` 只有版本，没有 `--hash`。要哈希得换成 `pip-compile --generate-hashes` 或 `uv lock`，并且要决定只锁 Linux x86-64 还是多平台。
- **间接依赖没有测试守着**：测试只核对直接依赖的 16 个包装的是不是钉的版本；另外 16 个间接依赖靠安装时的 `-c requirements-lock.txt`，不带 `-c` 安装时它们会是当时最新的版本，没有测试会失败。
- **基准的出处里没有记环境**：评审建议在基准结果里记一份 `pip freeze` 的摘要。那要改 `scripts/benchmark.py` 写进结果文件的内容，不属于"算的东西不变"的这一步。
- **`StarletteDeprecationWarning`**（starlette 1.9.0 提示 `TestClient` 用的 httpx 将来要换成 httpx2）：现在两个都钉住了，不会自己变；迁移到 httpx2 是另一件事。
- **装了旧版 Node 时 `tests/test_web.py` 是失败而不是跳过**：没有改。跳过会让 211 个 node 测试悄悄不跑。
- **`test_code_fingerprint_covers_the_code_a_run_executes` 在不是 git 检出的代码树上失败**：没有改，见 CI 一节。
- **没人引用的定义**（`stylize.py` 的 3 个）：下一步。
- **测试文件互相导入**（14 个文件、33 条）：只动了并行用得上的一处（真实服务进程的启动代码合成一份）；其余留给整理测试的那一步。
- **把演示脚本那个测试从慢测试里拿出来**（评审的建议）：没有做，它在 `tests/test_e2e.py` 里，和用 COLMAP 的夹具在同一个模块。
