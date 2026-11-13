# 配置

> 这份文档由 README 的"配置"一节和"项目结构"里数据目录那一段搬来（D34 C 部分第 2 步，2026-11-13），又补上了 `Params` 的全表。
> 两张大表里，"字段""默认值""还能怎么设"几列由 `scripts/gen_docs.py` 从代码生成，"含义"一列手写。方向是**代码为准，文档跟着**：代码里加一个字段、删一个字段、改一个默认值而文档没有跟着改，`tests/test_docs.py`（快速测试）就失败；改完代码运行 `python scripts/gen_docs.py` 更新这几列，新字段的"含义"要手写，空着测试也不过。
> "含义"里要写竖线时写成 `\|`，放在反引号里也一样（比如 `Path \| None`）：在表格里竖线就是格子的分界，不这样写，这一行会多出一格。遇到格子数和表头对不上的行，脚本不改写任何文档，退出码是 1，并指出是哪张表的哪一行；测试也在同一行上失败。

## 怎么改配置

产品阈值在 `crowd3d/config.py` 的 `Settings` 里。命令行只开放了数据目录和重建器两项，其余字段在代码里构造 `Settings(...)` 时修改。环境变量 `CROWD3D_STRUCTURE_CATALOG` 可以换一个地标目录，设成空值就关掉风格化作业；`CROWD3D_KEEP_WORK_DIRS=1`（或 `true`、`yes`、`on`）保留每个作业的工作目录（D34）。

合起来，一个 `Settings` 字段最多有四种设法，下面的表逐个字段列出后三种：

- **在代码里**构造 `Settings(字段=值)`：每个字段都可以。
- **环境变量**：只有四个字段有（`$CROWD3D_DATA_DIR`、`$CROWD3D_RECONSTRUCTOR`、`$CROWD3D_KEEP_WORK_DIRS`、`$CROWD3D_STRUCTURE_CATALOG`），在构造 `Settings()` 的那一刻读。
- **命令行**：`python -m crowd3d.api` 只有 `--data-dir` 和 `--reconstructor` 两项对应字段（`--host`、`--port`、`--seed-demo` 不是 `Settings` 的字段）。
- **基准配置**：`eval/configs/*.json` 的 `settings` 可以改标着"基准配置"的字段，别的字段基准不让改（见下面"其他参数"的第二条）。

其他参数：
- `crowd3d/params.py` 的 `Params`：算法内部参数，例如 GPS 和罗盘的 σ、RANSAC 门槛。和这一轮有关的几项：
  - `feature_cache`（默认 True）：地标特征缓存；
  - 地理配准怎么权衡 GPS 和罗盘（D34）：`gps_span_model_side`（True）、`compass_other`（`ignored`；另两个取值是 `trusted` 和 `weak`，后者用 `compass_other_sigma_deg` = 45°）、`yaw_arbitration`（False，阈值 `yaw_arb_min_inliers` 8、`yaw_arb_gps_sigmas` 5、`yaw_arb_min_readings` 3、`yaw_arb_max_scatter` 2）。前两个是在记录的运行上离线重放以后打开的：按事先写下的规则，它们俩是 dev 上合格的组里开关最少的，留出集上通过；第三个按规则选不到，在留出集上也没有起过作用，所以关着（[docs/BENCHMARK.md](BENCHMARK.md) 6.13 节）。前两个之后又用一次全新的留出集运行确认过，通过，保持默认（D34 第 7 步，6.14 节）；旧的行为是 False、`trusted`、False；
  - `map_continue`（默认 False）：从目标上一个模型续建，开发集验证没过；`map_rebuild_ratio`（1.5）、`map_max_continuations`（4）等是它的规则；
  - 地理锚点的下限：`anchor_yaw_floor_deg` 7.35°、`anchor_scale_floor` 0.352（log 尺度）、`anchor_xy_floor_m` 3.1 m、`anchor_tilt_floor_deg` 1.94°，高度 σ `anchor_z_sigma_m` 1.5 m，这组下限叫 `anchor_calibration_id` = `dev-core5-v1`。
- 基准的 `eval/configs/*.json` 只写要改的字段，没写的保持默认值。可以改的有：重建器、`Params`，以及 `Settings` 的产品阈值。不能改的是数据放在哪、进程怎么跑、产品自己的风格化步骤和只供显示的字段：`data_dir`、`reconstructor`、`inline_jobs`、`keep_work_dirs`、`max_job_attempts`、`max_concurrent_decodes`、`malloc_mmap_threshold_bytes`、`stylize_published`、`structure_catalog`、`structure_bindings`、`stylize_timeout_s`、`calibrated_anchor_ids`。
- 风格化和结构优先各模块的参数，都在代码里：
  - `crowd3d/stylize.py` 的 `StyleParams`（默认 `DEFAULT_PARAMS`），包括 #73 的三个开关 `snap_use_anchor`、`snap_refuse_clipped_window`、`snap_max_gps_median_m`（20 m），默认都开；
  - `crowd3d/structure_pipeline.py` 的 `Limits`（每个模型 1200 s、4096 MB、最多判 60 个分件，D25 `auto`，掩码只有 A+R）；
  - `crowd3d/structure.py` 的 `StructureParams`（`DEFAULT_STRUCTURE`）；
  - `crowd3d/relief.py` 的 `ReliefParams`（`DEFAULT_RELIEF`）；
  - `crowd3d/lines3d.py` 的 `LinesParams`（`DEFAULT_LINES`）。

## `Settings`（`crowd3d/config.py`）

<!-- gen_docs:begin settings -->
| 字段 | 默认值 | 除了在代码里，还能怎么设 | 含义 |
|---|---|---|---|
| `data_dir` | `data` | `$CROWD3D_DATA_DIR`、`--data-dir` | 数据目录 |
| `num_sectors` | `8` | 基准配置 | 扇区数，0 号以正北为中心，顺时针编号 |
| `face_min_photos` | `4` | 基准配置 | 一个扇区有多少张合格照片才生成立面模型 |
| `face_rebuild_growth` | `3` | 基准配置 | 立面模型每多几张照片重建一次 |
| `face_margin_deg` | `10.0` | 基准配置 | 立面作业额外使用相邻扇区多少度以内的照片 |
| `full_min_photos` | `24` | 基准配置 | 生成全景模型需要的照片总数和就绪扇区数 |
| `full_min_sectors` | `6` | 基准配置 | 见上一行：生成全景模型要有这么多个就绪扇区 |
| `full_reachability_min_photos` | `48` | 基准配置 | 照片达到这个数以后，仍然没有照片的方向算作"拍不到"，不再等它 |
| `full_rebuild_growth` | `8` | 基准配置 | 全景模型每多几张照片重建一次 |
| `default_radius_m` | `150.0` | 基准配置 | 新建地标的默认打卡半径 |
| `max_heading_offset_deg` | `60.0` | 基准配置 | 指南针朝向偏离地标方向的上限 |
| `facing_away_warns_other_refs` | `False` | 基准配置 | 打开后，没有标成真北（EXIF `GPSImgDirectionRef` 是 `M` 或没写）的朝向背对建筑时只警告（`heading_away_unverified`），不拒绝。这样收下的照片，重建时当作没有指南针（`validation.believed_heading`），质量门改查它估计出来的朝向。D34 加的，默认关。用全新的重建在 dev 划分上量过（D34 第 7 步，[docs/BENCHMARK.md](BENCHMARK.md) 6.14 节）：每次运行多收 12 张照片（460 → 472），但没有通过事先写下的守卫（两栋楼上离参考超过 20 m 的相机变多，多出来的都不是新收下的照片），所以没有在留出集上跑，保持关 |
| `min_image_side_px` | `480` | 基准配置 | 照片短边的最小像素数 |
| `blur_min_laplacian_var` | `30.0` | 基准配置 | 模糊度阈值（在长边 ≤ 800 px 的灰度图上计算） |
| `duplicate_max_hamming` | `4` | 基准配置 | dHash 距离不超过此值即判为重复 |
| `max_upload_bytes` | `26_214_400（25 MiB）` | 基准配置 | 单个文件的大小上限 |
| `max_image_pixels` | `50_000_000` | 基准配置 | 像素上限，超过判为 `too_large`。管所有按原尺寸解码的图：PNG、WebP，以及渐进式、分量分开扫描、无损或算术编码的 JPEG（解码器要把整幅图留在内存里，9000 × 9000 实测 290–525 MB）。只有顺序式、一次扫描的 JPEG 不受限制（相机和手机写的都是这种，能按 1/8 缩小解码，同样大小实测 104 MB），D34 |
| `max_concurrent_decodes` | `2` | — | 整个进程里同时让 Pillow 处理的上传个数，其余的排队（D34）：读文件头和 EXIF、解码像素都算。贴着像素上限的一次解码实测要 0.33 GB（渐进式 JPEG）到约 0.9 GB（WebP）。在真实的服务上实测（同时 8、16、40 个上传）：PNG 0.6 GB、WebP 1.6 GB，空闲时回到 0.1 GB；这要靠下一行的设置才成立。排队的上传各自拿着自己的文件（最多 25 MiB） |
| `malloc_mmap_threshold_bytes` | `4_194_304（4 MiB）` | — | 大于 0 时服务从启动起管住 glibc 的分配器（D34），两件事。一是把 mmap 阈值固定在这个值：这么大或更大的内存块释放时直接还给操作系统；不固定的话，每个解码过大图的工作线程都留着几百 MB，8 个 PNG 同时上传实测 2.5–3.4 GB，空闲也不还。二是每个解码名额结束时把释放掉的内存还给操作系统（`malloc_trim`），管更小的块：一个 69 KB 的 PNG 带 64 MiB 压缩文本，16 个、再 40 个同时上传之后，空闲的服务曾留着 2.3 GB。0 表示不动分配器，C 库不是 glibc 时也不动；环境变量 `MALLOC_MMAP_THRESHOLD_` 已经设了时阈值不动，只做第二件 |
| `subject_max_distance_m` | `50.0` | 基准配置 | 照片资格（D31）：相机 GPS 离建筑 OSM 主轮廓不超过这个距离（米），照片就合格。设得很大等于关掉这条规则 |
| `subject_min_share` | `0.33` | 基准配置 | 照片资格（D31）：离得更远时，估算建筑占画面（宽或高）的比例要到这个数，否则判 `not_subject`。取 1/3 而不是 1/2，是因为估算偏低：拿有参考位姿的照片核对，1/3 最接近建筑真的占了半个画面 |
| `subject_default_focal_35mm` | `28.0` | 基准配置 | 照片资格估算视场用的 35 mm 等效焦距：EXIF 没有或不在 [10, 1200] mm 时按 28 mm 算 |
| `subject_min_focal_35mm` | `10.0` | 基准配置 | 见上一行：EXIF 的 35 mm 等效焦距低于它就不信，按默认焦距算 |
| `subject_max_focal_35mm` | `1200.0` | 基准配置 | 见 `subject_default_focal_35mm`：EXIF 的 35 mm 等效焦距高于它就不信，按默认焦距算 |
| `processing_max_side_px` | `1600` | 基准配置 | 送入 SfM 的处理图长边上限 |
| `reconstructor` | `'colmap'` | `$CROWD3D_RECONSTRUCTOR`、`--reconstructor` | `colmap` 或 `fake` |
| `camera_height_m` | `1.6` | 基准配置 | 假定的拍摄高度（手机的 GPS 高度太不准，不用） |
| `inline_jobs` | `False` | — | 在请求内同步执行作业（测试用） |
| `keep_work_dirs` | `False` | `$CROWD3D_KEEP_WORK_DIRS` | 保留每个作业的工作目录：重建作业的 COLMAP 数据库和模型（每张照片约 1.5 MB），风格化作业的 `spec.json`、`child.log` 和输出；风格化脚本的 `--job-dir` 需要 |
| `max_job_attempts` | `2` | — | 同一个作业最多尝试几次：进程崩溃中断它这么多次后，判为失败，不再重跑 |
| `stylize_published` | `True` | — | 发布的模型自动排风格化作业（地标要有分件快照） |
| `structure_catalog` | `eval/landmarks.json` | `$CROWD3D_STRUCTURE_CATALOG` | 带分件快照的地标目录 |
| `structure_bindings` | `{}` | — | 地标 id → 目录 slug，给 id 不等于 slug 的地标用（目录中心仍要在地标半径内） |
| `stylize_timeout_s` | `1200.0` | — | 风格化子进程的时间上限，超时就杀掉，这个模型没有 GLB |
| `localize_photos` | `False` | 基准配置 | 视觉定位作业（D30）：把公开的（D32 起是通过、没撤回的）、但没有显示中的 SfM 相机的照片用 PnP 定位到地标的已存地图里，只用于显示（`/poses` 的 `loc`）。上传请求里只做便宜的判断，作业在没有重建作业排队时才跑，有重建作业排队就在两张照片之间停下。D30 的验证里代价规则不过（定位用掉运行 CPU 的 0.267 和 0.665，上限 0.25；dev 那一条落在机器速度的漂移之内，稳的是 test 那一条），所以保持关闭，见 [D30 的记录](decisions/D30-photo-poses.md)。关掉后以前存下的 `loc` 位姿马上都不显示（记录留着，再打开时照旧显示），发现定位错了就靠它撤下 |
| `localize_batch` | `50` | 基准配置 | 一个定位作业最多试几张照片，其余的留给下一个作业 |
| `calibrated_anchor_ids` | `()` | — | 通过留出集标定的锚点标定 id；只有列在这里，`ModelInfo.geo.calibrated` 才为 true、地标页才显示"位置 ±X 米（1σ）" |
| `qa_min_registered_fraction` | `0.6` | 基准配置 | 发布质量门：立面 / 全景模型的最低注册率 |
| `qa_full_min_registered_fraction` | `0.7` | 基准配置 | 见上一行：全景模型的最低注册率 |
| `qa_face_min_registered` | `6` | 基准配置 | 发布质量门（D31）：立面注册率不够时，注册这么多张、并且地理配准拟合上了 GPS，注册这一项也算过；其余检查不变，全景模型不适用 |
| `qa_min_points` | `300` | 基准配置 | 发布质量门：最少点数 |
| `qa_max_reprojection_error_px` | `2.5` | 基准配置 | 发布质量门：平均重投影误差的上限 |
| `qa_max_twin_fraction` | `0.1` | 基准配置 | 发布质量门：双胞胎相机（GPS 和罗盘都对不上）的比例上限 |
| `qa_max_flagged_fraction` | `0.2` | 基准配置 | 发布质量门：带任何可疑标记的相机比例上限 |
| `qa_max_compass_error_deg` | `25.0` | 基准配置 | 发布质量门：各相机与罗盘偏差的中位数上限 |
<!-- gen_docs:end settings -->

## `Params`（`crowd3d/params.py`）

算法内部的参数，产品运行时用的就是下面的默认值。它们没有环境变量，也没有命令行参数，只有两种设法：基准配置（`eval/configs/*.json` 的 `params`，每个字段都可以改，写错名字或类型会报错），或者在代码里用 `params.override(字段=值)`、`params.activate(Params(...))`。参数机制的说明在 [BENCHMARK.md](BENCHMARK.md) 4.2 节。

<!-- gen_docs:begin params -->
| 字段 | 默认值 | 含义 |
|---|---|---|
| `gps_sigma_m` | `6.0` | 照片没报 GPS 精度时，每个轴的 1σ（米）。真实手机在建筑旁边的定位约偏 10 m（洗礼堂的评测） |
| `gps_sigma_min_m` | `2.0` | 照片自己报的精度（`GPSHPositioningError`）的下限（米） |
| `gps_inlier_sigmas` | `3.0` | 地理配准的 RANSAC 内点阈值是 max(这个数 × σ, `gps_inlier_min_m`) |
| `gps_inlier_min_m` | `10.0` | 见上一行：内点阈值的下限（米） |
| `max_gps_scale_error` | `0.35` | GPS 拟合预计留下的相对尺度误差（等于偏航误差的弧度数）的上限，超过就不算用 GPS 配准上 |
| `compass_sigma_deg` | `15.0` | 一个罗盘读数的 1σ（度）：手机罗盘在钢筋混凝土旁边很差 |
| `compass_yaw_min_sigma_deg` | `3.0` | 罗盘给出的偏航 σ 的下限：同一个地点的读数受同样的磁干扰 |
| `compass_outlier_sigmas` | `3.0` | 算罗盘偏航时，离平均值这么多个 σ 的读数丢掉 |
| `pseudo_compass_sigma_deg` | `25.0` | "伪罗盘"（假设每张照片都朝地标中心拍）一个读数的 1σ（度） |
| `pseudo_compass_min_sigma_deg` | `10.0` | 伪罗盘偏航 σ 的下限：拍立面的照片对着建筑的一部分，不是中心 |
| `max_yaw_conflict_deg` | `40.0` | GPS 拟合的偏航和罗盘（或伪罗盘）的偏航相差超过它，就不信 GPS 的偏航 |
| `yaw_systematic_sigma_deg` | `3.0` | 报出的偏航 σ 再按平方和加上这个数：同一个地点的定位和读数共有的误差；吸附的偏航窗口用它 |
| `gps_span_model_side` | `True` | 打开："分布"门和 GPS 的尺度、偏航 σ 在拟合出的相机布局上量（SfM 相机中心乘拟合出的尺度），不在带噪声的 GPS 坐标上量。挤在一个点周围的定位自己也有散布，老规则因此放过了 15–44% 这样的照片集，尺度差 2–4 倍。D34 打开，见下面"靠测量定下来的开关" |
| `compass_other` | `'ignored'` | 没标成真北的罗盘朝向（EXIF `GPSImgDirectionRef` 是 `M` 或没写）怎么用：`trusted` 和真北的一样用；`weak` 算一个 σ 为 `compass_other_sigma_deg` 的读数；`ignored` 不算读数。偏航先验、逐台相机的罗盘标记、候选排序和重试、质量门的罗盘检查都按它。D34 改成 `ignored`，见下面"靠测量定下来的开关" |
| `compass_other_sigma_deg` | `45.0` | `compass_other` 是 `weak` 时那种读数的 1σ（度），必须是正数 |
| `yaw_arbitration` | `False` | 打开：偏航冲突（见 `max_yaw_conflict_deg`）按两边的强弱裁决，不再一律归罗盘先验。GPS 拟合强而罗盘先验弱时跟 GPS 走，这时先验不参加融合，它的读数也不用来标记相机、不进质量门的罗盘检查；别的冲突照旧归先验（伪罗盘也算）。默认关，见下面"靠测量定下来的开关"。四个阈值来自评审和模拟，不是真实数据 |
| `yaw_arb_min_inliers` | `8` | 偏航裁决：GPS 拟合至少有这么多个内点才算强 |
| `yaw_arb_gps_sigmas` | `5.0` | 偏航裁决：冲突至少是 GPS 自己偏航 σ 的这么多倍，GPS 拟合才算强 |
| `yaw_arb_min_readings` | `3.0` | 偏航裁决：彼此一致的读数（折合成真北读数）少于这么多个，罗盘先验算弱 |
| `yaw_arb_max_scatter` | `2.0` | 偏航裁决：读数的散布超过自己 σ 的这么多倍，罗盘先验算弱 |
| `up_prior_weight` | `0.01` | 估计重力方向：相机"上"向量平均值（带仰拍偏差）的拉力，只用来定正负号 |
| `up_min_eigen_gap` | `0.02` | 估计重力方向：最小的两个特征值相差不到它时，倾斜量不出来，直接用"上"向量的平均值 |
| `up_common_height_fraction` | `0.25` | 估计重力方向：相机离中位高度不超过"这个数 × 相机的水平散布"才算站在同一高度；更高处（塔上、露台上）拍的不用来定地面 |
| `anchor_yaw_floor_deg` | `7.35` | 地理锚点偏航 σ 的下限（度）。下限按平方和加到每个分量自己的 σ 上（同一个地点的定位共有的误差）；这一组叫 `dev-core5-v1`，只在开发集上拟合 |
| `anchor_scale_floor` | `0.352` | 地理锚点尺度 σ 的下限，按对数尺度（尺度误差有重尾：圣母教堂、勃兰登堡门） |
| `anchor_xy_floor_m` | `3.1` | 地理锚点水平位置 σ 的下限（米），每个水平轴，在支点处 |
| `anchor_tilt_floor_deg` | `1.94` | 地理锚点倾斜 σ 的下限（度），每个水平轴 |
| `anchor_z_sigma_m` | `1.5` | 地理锚点的高度 σ（米）：高度来自"拍摄高度"的假设 |
| `anchor_fallback_log_scale_sigma` | `0.7` | 没有可用的 GPS 拟合时，粗估尺度的对数 σ（粗估的尺度实测偏大 1.3–2.4 倍） |
| `anchor_calibration_id` | `'dev-core5-v1'` | 这组下限的名字。名字列在 `Settings.calibrated_anchor_ids` 里，API 才说"已标定" |
| `exhaustive_max_images` | `60` | 不超过这么多张时做穷举匹配，更多时按 GPS 做空间匹配 |
| `spatial_max_neighbors` | `30` | 空间匹配：每张照片最多配这么多个邻居 |
| `spatial_max_distance_m` | `80.0` | 空间匹配：只配这么远（米）以内的照片，也就是拍同一侧的 |
| `focal_refine_min_images` | `12` | 少于这么多张时固定用 EXIF 焦距，不优化 |
| `mapping_attempts` | `3` | 不超过 `small_set_images` 张的照片集，建图一律做这么多次（换种子） |
| `mapping_attempts_large` | `2` | 更大的照片集最多建这么多次，而且只有结果可疑才再建 |
| `small_set_images` | `30` | "小照片集"最多这么多张 |
| `min_candidate_points` | `100` | 建出来的模型点数少于它算退化 |
| `feature_cache` | `True` | 地标特征缓存：作业开始前把缓存里已有的特征和匹配写进自己的数据库，COLMAP 只算新的。默认开，见下面"靠测量定下来的开关" |
| `map_continue` | `False` | 从目标正在显示的那个模型续建（持久地图第 1b 步），不从零建图。默认关，见下面"靠测量定下来的开关" |
| `map_rebuild_ratio` | `1.5` | 续建：照片数达到上一次从零建图时的这么多倍，就从零重建 |
| `map_max_continuations` | `4` | 续建：连着续建这么多次以后，从零重建 |
| `map_fix_existing` | `False` | 续建：打开后，新照片注册时底模的相机固定不动 |
| `map_continue_min_kept` | `0.95` | 续建：续建出来的模型至少要保住底模这么大比例的已注册照片 |
| `retry_flagged_fraction` | `0.05` | 和自己的传感器对不上的相机超过这个比例，结果算可疑，再建一次 |
| `retry_compass_error_deg` | `20.0` | 或者平均罗盘偏差（封顶 90°）超过它，结果也算可疑 |
| `gross_compass_sigmas` | `3.0` | 逐台相机的检查：和自己的罗盘相差这么多个罗盘 σ，算错，不算噪声 |
| `far_from_gps_factor` | `3.0` | 逐台相机的检查：离自己的 GPS 这么多倍内点阈值，算深度错了或者立面错了 |
| `focal_ratio_min` | `0.25` | 逐台相机的检查：焦距 / 图像长边 在 [min, max] 之外算退化的相机 |
| `focal_ratio_max` | `12.0` | 见上一行：上限 |
| `analysis_max_side_px` | `800` | 校验：模糊度和 dHash 在缩到这个长边的副本上算 |
| `qa_max_facing_offset_deg` | `90.0` | 质量门：没有罗盘的照片，估计出来的朝向偏离"相机 → 地标"方向这么多度就标记 |
| `qa_full_sector_min_registered` | `2` | 质量门：一个就绪扇区里有这么多张注册的照片，才算全景模型覆盖到了它 |
| `loc_min_inliers` | `30` | 视觉定位（D30）：至少要这么多个不同的 2D–3D 对应，PnP 的内点也至少这么多 |
| `loc_min_inlier_ratio` | `0.75` | 视觉定位：内点 / 对应 的下限（认错面的到过 0.67–0.68，对的在 0.72 以上） |
| `loc_max_reproj_frac` | `0.0025` | 视觉定位：内点的平均重投影误差不超过"这个数 × 图像长边" |
| `loc_ransac_frac` | `0.012` | 视觉定位：PnP 的 RANSAC 阈值是"这个数 × 图像长边" |
| `loc_second_ratio` | `0.15` | 视觉定位：离最好的位姿很远的第二个位姿，内点数超过最好那个的这个比例，就算有歧义（认错面的 0.18 以上，对的 0.12 以下） |
| `loc_second_m` | `5.0` | 见上一行，"很远"指相距这么多米（局部距离乘锚点的尺度） |
| `loc_second_deg` | `10.0` | 或者转了这么多度 |
| `loc_focal_mismatch` | `0.2` | 视觉定位：放开焦距核对出来的焦距和 EXIF 焦距相差（相对值）超过它，不给位姿 |
| `loc_min_focal_ratio` | `0.3` | 视觉定位：EXIF 焦距和核对出来的焦距，除以图像长边，都要在 [min, max] 里 |
| `loc_max_focal_ratio` | `8.0` | 见上一行：上限 |
| `loc_across_half` | `0.5` | 视觉定位：另一张地图也接受，内点数达到最好那张的这个比例 |
| `loc_across_deg` | `25.0` | 而且朝向相差这么多度：判为 `ambiguous_across_models` |
| `loc_min_height_m` | `-15.0` | 视觉定位：相机高度（ENU 的上）的下限（米） |
| `loc_max_height_m` | `120.0` | 视觉定位：相机高度的上限（米） |
| `loc_max_look_deg` | `75.0` | 视觉定位：光轴和"相机 → 模型点中位处"方向的夹角上限（度） |
| `loc_max_pairs` | `200` | 视觉定位：一张查询照片最多和地图里这么多张照片匹配，超过时取 GPS 最近的 |
| `loc_withhold_gps_conflict` | `False` | 视觉定位：打开后，被自己的 GPS 像 SfM 相机那样标记的位姿不给出；关着只记录 |
<!-- gen_docs:end params -->

## 靠测量定下来的开关

下面这些字段的默认值不是拍脑袋定的，是按事先写下的规则在测试集上量出来的。要改它们，先看对应的记录，再重新量。

| 字段 | 现在的默认 | 量出来的结论 | 记录 |
|---|---|---|---|
| `Params.feature_cache` | 开 | 建图读到的数据库和不用缓存时逐字节相同（验证了 46 个作业），核心五栋的作业 CPU 少了 59.7% | [BENCHMARK.md 6.7 节](BENCHMARK.md#67-地标特征缓存的验证持久地图第-1a-步) |
| `Params.map_continue` | 关 | 开发集验证没过：版本连续性 56%（要求 90%）；先试续建再从零重建，净省的 CPU 为零 | [BENCHMARK.md 6.8 节](BENCHMARK.md#68-续建的验证持久地图第-1b-步) |
| `Params.gps_span_model_side`、`Params.compass_other` | 开、`ignored` | 先在记录的运行上离线重放：按事先写下的规则，它们俩是 dev 上合格的组里开关最少的，留出集通过；再用一次全新的留出集运行确认，通过 | [BENCHMARK.md 6.13 节](BENCHMARK.md#613-d34-的验证地理配准怎么权衡-gps-和罗盘离线重放)、[6.14 节](BENCHMARK.md#614-d34-的全新运行第四个开关和地理配准默认值的全新确认)，[D34 的记录](decisions/D34-review-fixes.md) |
| `Params.yaw_arbitration` | 关 | 按规则选不到，在留出集上也没有起过作用 | [BENCHMARK.md 6.13 节](BENCHMARK.md#613-d34-的验证地理配准怎么权衡-gps-和罗盘离线重放) |
| `Settings.facing_away_warns_other_refs` | 关 | dev 的全新运行每次多收 12 张照片（460 → 472），但没有通过事先写下的守卫，所以没有在留出集上跑 | [BENCHMARK.md 6.14 节](BENCHMARK.md#614-d34-的全新运行第四个开关和地理配准默认值的全新确认) |
| `Settings.localize_photos` | 关 | D30 的验证里代价规则不过（定位用掉运行 CPU 的 0.267 和 0.665，上限 0.25），准确性和隐私的守卫都过 | [BENCHMARK.md 6.12 节](BENCHMARK.md#612-d30-的验证视觉定位同一批产物上的配对补算)，[D30 的记录](decisions/D30-photo-poses.md) |
| `Settings.subject_max_distance_m`、`Settings.subject_min_share`、`Settings.qa_face_min_registered` | 50 m、0.33、6 | D31 的全新运行按事先写好的标准是 FAIL（dev PASS，test FAIL）。规则现在开着，留不留、阈值取多少等产品负责人决定 | [BENCHMARK.md 6.11 节](BENCHMARK.md#611-d31-的验证照片资格和立面按张数过门)，[D31 的记录](decisions/D31-photo-eligibility.md) |
| `Settings.calibrated_anchor_ids` | 空（关） | 现在这组锚点下限在留出集上的标定没有通过：先是证据不足，D31 的全新运行里四个分量的 2σ 覆盖都过，只有"水平位置在 3σ 外的比例"一项不过。所以地标页不显示"位置 ±X 米（1σ）" | [BENCHMARK.md 7.8 节](BENCHMARK.md#78-地理误差覆盖标定版本第-7-版)，[REVIEW_ADOPTION.md](REVIEW_ADOPTION.md) |
| `Params.anchor_yaw_floor_deg` 等四个下限（`dev-core5-v1`） | 见上表 | 只在开发集上拟合（样本内，约 4 栋有效建筑），是候选值，不是结论 | [BENCHMARK.md 6.6 节](BENCHMARK.md#66-标定表和下限拟合calibration) |

风格化的吸附门还有三个量过的开关（任务 #73 的 `snap_use_anchor`、`snap_refuse_clipped_window`、`snap_max_gps_median_m`），它们在 `StyleParams` 里，不在上面两张表里（见"怎么改配置"的最后一条）。

## 数据目录

数据目录（`Settings.data_dir`，默认是 `./data`）：

```
photos/<地标>/<照片>.jpg       原图（只保存通过校验的照片）
photos/<地标>/<照片>.proc.jpg  处理图：已按 EXIF 方向转正，长边 ≤ 1600，EXIF 只保留焦距和机型；照片接口（D32）每次从它重新编码出去掉元数据的公开照片，不另存
models/<地标>/<模型>.ply/.json 模型和元数据
models/<地标>/<模型>.glb       风格化作业的 GLB（旁边的 .style.json 是它的摘要，不对外提供）
models/<地标>/<模型>.stylize-failed.json  风格化作业失败时留下的原因（D34）：错误、子进程的结果、它输出的最后 8 KiB；QA 不过时旁边还有
                               <模型>.stylize-failed.style.json（完整的 QA：哪道门、哪个分件）。都不对外提供，之后风格化成功就删掉
maps/<地标>/cache.sqlite3      特征和匹配缓存（不淘汰，每张照片约 1.1–1.7 MB）
maps/<地标>/models/<模型>/     已发布 COLMAP 模型的稀疏模型和 map.json（每个目标留最新 2 个）
maps/<地标>/structure-cache/   风格化作业的照片缓存（天空掩码、线段）
maps/<地标>/subject-<sha12>.npz  照片资格用的建筑采样点（按分件快照的 sha256 命名，D31）
work/<作业>/                   COLMAP 工作目录；风格化作业用 work/<作业>.<第几次>/
crowd3d.sqlite3
public_id.salt                 公开 id 和撤回凭证的密钥（D30，权限 0600），要和数据库一起备份、一起拷贝
```

不合格的上传只在数据库里留下记录（原因、位置），不保存图片。文件名一律用 uuid，不使用客户端传来的文件名。`maps/` 下的东西都不对外提供。

## 服务要在仓库的检出目录里运行

这是现在的限制，写在这里免得踩到（评审"工程做法"的一条，D34 C 部分只补了文档，没有改代码）：

- **包装不了。** `pyproject.toml` 没有 `[build-system]`，`pip install .` 装不出能用的包。要运行服务，就在 git 检出的根目录里执行 `python -m crowd3d.api`。
- **网页文件取自检出目录的 `web/`。** 服务提供的页面是 `crowd3d/` 旁边的 `web/` 目录（`crowd3d.api.WEB_DIR`），没有配置项。
- **默认的地标目录是评测目录里的文件。** `Settings.structure_catalog` 默认是检出目录的 `eval/landmarks.json`，OSM 轮廓和分件快照在 `eval/osm/` 下。所以改这几个评测文件，也就改了服务的行为：哪些地标出风格化模型，照片资格按哪个轮廓判。
- **目录里写的文件路径，相对"目录文件的上两级目录"解析。** 目录条目里的 `osm.file` 和 `osm_parts.file` 写的是 `eval/osm/<slug>.parts.osm` 这样的路径，代码把它接在 `目录文件所在目录的上一级` 后面（`crowd3d/catalog.py` 的 `find_source`：`Path(catalog).resolve().parent.parent`）。默认目录是 `<检出目录>/eval/landmarks.json`，上两级正好是检出目录。用 `$CROWD3D_STRUCTURE_CATALOG` 换一个目录时要照这个约定摆放：目录放在 `<某处>/<任意一层>/landmarks.json`，它引用的文件放在 `<某处>/` 下、和条目里写的相对路径一致。
- **摆错了会怎样。** 分件快照找不到、或者 sha256 和目录里记的不一样时，风格化作业失败，错误信息是 `<slug>: <路径> is missing`（或 `is not the snapshot the catalogue lists`）；照片资格检查读不出建筑时按设计放行，上传照收、不查"建筑是不是主体"，只在日志里留一条 warning。目录文件本身读不出来时，这个地标当作没有绑定建筑，日志里也是一条 warning。

把目录和快照移进包里、让包能安装、每个字段都能用环境变量设，这些是评审的建议，都还没有做（见 [STATUS.md](STATUS.md)）。
