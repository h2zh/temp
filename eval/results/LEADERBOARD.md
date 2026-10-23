# Crowd3D 基准排行榜（benchmark leaderboard）

由 `python -I scripts/benchmark.py report` 根据 `eval/results/<testset>/*.json` 生成，请勿手工编辑。指标定义见文末；方向：↑ 越大越好，↓ 越小越好。

## 全部运行

| 测试集 | 运行 | 配置 | 开始时间 (UTC) | 提交 | 建筑 成功/总数 | models_published (models) | models_qa_rejected (models) | facades_published (sectors) | full_model_published (0/1) | registration_rate (fraction) | estimate_coverage (fraction) | outline_rotation_abs_deg (deg) | outline_shift_m (m) | gt_rotation_error_median_deg (deg) | gt_gross_shown (cameras) | style_models_eligible (models) | style_registration_pass_rate (fraction) | style_registration_right (models) | style_registration_wrong (models) | style_rejected_placed_right (models) | style_walls_coloured (walls) | style_qa_failed (models) | wall_time_s (s) | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| commons-ten-v1 | v0-noise | default | 2026-10-23 05:36 | e264f7e | 2/2 | 8 | 11 | 6 | 0 | 0.653 | 0.459 | 3.5 | 9.85 | 0.237 | 0 | 7 | 0.429 | - | 0 | - | 2 | 0 | 1489 | v0 noise floor on real Commons photos: 3 repeats of the 2 cheapest core buildings (metric schema 2, stylize on). Not a performance record; v0-baseline is. |
| commons-ten-v1 | v0-baseline | default | 2026-10-23 05:42 | e264f7e | 5/5 | 54 | 55 | 19 | 0 | 0.662 | 0.598 | 3.82 | 10.1 | 0.744 | 7 | 51 | 0.608 | - | 19 | - | 16 | 0 | 9720 | v0 baseline on the CORE FIVE Commons buildings only (metric schema 2, style_* metrics, stylize on, keep_work_dirs). Left out of this round (the container restarts about every 3 h and a large building takes 45-50 min): sagrada-familia, hungarian-parliament, lincoln-memorial, india-gate, sydney-opera-house. Time cap: brandenburg-gate and frauenkirche-dresden hit the soft cap (no new upload after 80% of the 3600 s hard cap), so only 131/150 and 77/111 of their photos were uploaded. All five building runs are reused via --resume (same run key: code, config, test set, inputs, options); they ran 2026-10-23 01:46-03:44 UTC (four) and 04:44-05:36 UTC (frauenkirche-dresden). |
| commons-ten-v1 | v1-after-fixes | default | 2026-10-23 10:08 | 67b4301+dirty | 5/5 | 61 | 63 | 19 | 0 | 0.68 | 0.589 | 4.13 | 9.76 | 0.89 | 9 | 58 | 0.0862 | 5 | 0 | 10 | 0 | 0 | 9148 | v1 after the registration fixes (windowed snap STYLIZED_MODEL 3.2 with the model's yaw sigma + systematic floor, outline symmetry period, translation-only on near-circles, unclamped multi-start ICP with the 25 m shift bound measured at the walls, shift-scan ambiguity refusal, GPS depth check, scale-sigma and outline-rescale refusals; metric schema 3), stylize on, keep_work_dirs. CORE FIVE only, as v0-baseline; left out: sagrada-familia, hungarian-parliament, lincoln-memorial, india-gate, sydney-opera-house (about 2.5 h of COLMAP for the core five alone; the container restarts about every 3 h). |
| mdx-two-v1 | default-noise | default | 2026-10-22 01:48 | 94ee607 | 2/2 | 4 | 1 | 4 | 0 | 0.673 | 0.558 | 26.4 | 9.84 | 0.351 | 0 | - | - | - | - | - | - | - | 58.5 | noise floor: the baseline five times on both buildings |
| mdx-two-v1 | fast-1 | fast | 2026-10-22 01:53 | 94ee607 | 2/2 | 3 | 1 | 3 | 0 | 0.577 | 0.475 | 1.87 | 7.85 | 0.431 | 0 | - | - | - | - | - | - | - | 56.2 | fast config (1024 px); these photos are 1000 px, below both caps, so the config changes nothing here: the differences to default are run-to-run variation |
| mdx-two-v1 | v0-baseline | default | 2026-10-23 01:31 | 99c006c+dirty | 2/2 | 4 | 1 | 4 | 0 | 0.673 | 0.558 | 3.52 | 14.2 | 0.551 | 0 | 2 | 0.5 | - | 0 | - | 2 | 0 | 76.8 | v0 baseline, metric schema 2 (style_* metrics), stylize on, keep_work_dirs |
| mdx-two-v1 | v0-noise | default | 2026-10-23 01:34 | 99c006c+dirty | 2/2 | 4 | 1 | 4 | 0 | 0.673 | 0.558 | 12.9 | 11.8 | 0.655 | 0 | 2 | 0.5 | - | 0 | - | 2 | 0 | 86.1 | v0 noise floor (5 repeats), metric schema 2 (style_* metrics), stylize on, keep_work_dirs |
| mdx-two-v1 | v1-after-fixes | default | 2026-10-23 10:07 | 67b4301+dirty | 2/2 | 3 | 1 | 3 | 0 | 0.577 | 0.45 | 2.39 | 11 | 0.346 | 0 | 2 | 0.5 | 1 | 0 | 0 | 2 | 0 | 59.9 | v1 after the registration fixes (windowed snap STYLIZED_MODEL 3.2 with the model's yaw sigma + systematic floor, outline symmetry period, translation-only on near-circles, unclamped multi-start ICP with the 25 m shift bound measured at the walls, shift-scan ambiguity refusal, GPS depth check, scale-sigma and outline-rescale refusals; metric schema 3), stylize on, keep_work_dirs |
| synthetic-v1 | default-1 | default | 2026-10-22 01:45 | 94ee607 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.58 | 1.16 | 0.194 | 0 | - | - | - | - | - | - | - | 59.8 | end-to-end validation: run 1 of 2 identical runs |
| synthetic-v1 | default-2 | default | 2026-10-22 01:46 | 94ee607 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.58 | 1.16 | 0.194 | 0 | - | - | - | - | - | - | - | 59.4 | end-to-end validation: run 2 of 2 identical runs |
| synthetic-v1 | fast-1 | fast | 2026-10-22 01:47 | 94ee607 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.57 | 1.17 | 0.2 | 0 | - | - | - | - | - | - | - | 58.4 | fast config (1024 px); these photos are 640 px, below both caps, so the config changes nothing here: the differences to default are run-to-run variation |
| synthetic-v1 | v0-baseline | default | 2026-10-23 01:25 | 99c006c+dirty | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.57 | 1.17 | 0.199 | 0 | 12 | 0.5 | - | 2 | - | 6 | 0 | 78.7 | v0 baseline, metric schema 2 (style_* metrics), stylize on, keep_work_dirs |
| synthetic-v1 | v0-noise | default | 2026-10-23 01:27 | 99c006c+dirty | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.57 | 1.17 | 0.199 | 0 | 12 | 0.5 | - | 2 | - | 6 | 0 | 74.1 | v0 noise floor (3 repeats), metric schema 2 (style_* metrics), stylize on, keep_work_dirs |
| synthetic-v1 | v1-after-fixes | default | 2026-10-23 10:06 | 67b4301+dirty | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.57 | 1.17 | 0.199 | 0 | 12 | 0.333 | 4 | 0 | 4 | 6 | 0 | 68.5 | v1 after the registration fixes (windowed snap STYLIZED_MODEL 3.2 with the model's yaw sigma + systematic floor, outline symmetry period, translation-only on near-circles, unclamped multi-start ICP with the 25 m shift bound measured at the walls, shift-scan ambiguity refusal, GPS depth check, scale-sigma and outline-rescale refusals; metric schema 3), stylize on, keep_work_dirs |

## 每个配置最新一次运行的分建筑结果

### commons-ten-v1 · default · v1-after-fixes

| 建筑 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| florence-baptistery | ok | 51 | 6 | 1 | 5 | 0 | 0.946 | 0.745 | 1.6 | 10.8 | 0.29 | 6.54 | 2.25 | 0.237 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 1 | 1 | 3.07 | 0 | 10 | 0 | 235 | 8.7 |
| aachen-cathedral | ok | 75 | 2 | 10 | 1 | 0 | 0.36 | 0.173 | 8.2 | 10.4 | 0.56 | 7.7 | 2.92 | - | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 26.7 | 0 | 30 | 0 | 397 | 4 |
| brandenburg-gate | ok | 141 | 22 | 25 | 5 | 0 | 0.553 | 0.603 | 4.13 | 9.76 | 0.22 | 9.45 | 7.62 | 1.01 | 10 | 9 | 22 | 0 | 0 | 0 | 7 | 12 | 6 | 7.61 | 0 | 4 | 0 | 3162 | 22.3 |
| leaning-tower-pisa | ok | 118 | 11 | 17 | 3 | 0 | 0.712 | 0.652 | 3.61 | 6.79 | 0.22 | 8.85 | 6.1 | 1.27 | 0 | 0 | 11 | 1 | 1 | 0 | 2 | 8 | 0 | 6.59 | 0 | 8 | 0 | 2208 | 5.1 |
| frauenkirche-dresden | ok | 83 | 20 | 10 | 5 | 0 | 0.829 | 0.771 | 13.7 | 8.38 | 0.53 | 9.94 | 10.2 | 0.768 | 2 | 0 | 18 | 4 | 4 | 0 | 1 | 2 | 6 | 5.13 | 0 | 27 | 0 | 3144 | 55.3 |

### mdx-two-v1 · default · v1-after-fixes

| 建筑 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| florence-baptistery | ok | 20 | 3 | 0 | 3 | 0 | 0.821 | 0.9 | 2.39 | 11 | 0.33 | 6.55 | 2.45 | 0.346 | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 2.93 | 2 | 10 | 0 | 35.8 | 4.6 |
| aachen-cathedral | ok | 24 | 0 | 1 | 0 | 0 | 0.333 | 0 | - | - | - | - | - | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - | - | 0 | 24.1 | - |

### mdx-two-v1 · fast · fast-1

| 建筑 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| florence-baptistery | ok | 20 | 3 | 0 | 3 | 0 | 0.821 | 0.95 | 1.87 | 7.85 | 0.22 | 7.38 | 2.51 | 0.431 | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | 35.1 | - |
| aachen-cathedral | ok | 24 | 0 | 1 | 0 | 0 | 0.333 | 0 | - | - | - | - | - | - | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | 21.1 | - |

### synthetic-v1 · default · v1-after-fixes

| 建筑 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic-ring | ok | 32 | 9 | 0 | 8 | 1 | 1 | 1 | 2.19 | 1.4 | 0.03 | 2.56 | 4.98 | 0.259 | 0 | 0 | 9 | 3 | 3 | 0 | 3 | 1 | 0 | 0.915 | 4 | 4 | 0 | 60.8 | 11.5 |
| synthetic-two-sides | ok | 12 | 3 | 0 | 2 | 0 | 1 | 0.917 | 2.96 | 0.93 | 0.025 | 2.07 | 4.41 | 0.139 | 0 | 0 | 3 | 1 | 1 | 0 | 1 | 0 | 0 | 0.52 | 2 | 4 | 0 | 7.7 | 4 |

### synthetic-v1 · fast · fast-1

| 建筑 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic-ring | ok | 32 | 9 | 0 | 8 | 1 | 1 | 1 | 2.18 | 1.4 | 0.03 | 2.56 | 4.98 | 0.259 | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | 50.5 | - |
| synthetic-two-sides | ok | 12 | 3 | 0 | 2 | 0 | 1 | 0.917 | 2.96 | 0.93 | 0.025 | 2.07 | 4.41 | 0.14 | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | 7.9 | - |

## 运行间噪声（同一配置重复运行的极差）

- commons-ten-v1 · v0-noise（3 次，建筑 aachen-cathedral, florence-baptistery）：registration_rate 0.0178; outline_rotation_abs_deg 2.45; outline_shift_m 1.2; outline_residual_m 0.06; gps_residual_median_m 0.2; compass_error_median_deg 0.96; wall_time_s 593; jobs_time_s 590; style_registration_passed 2; style_registration_pass_rate 0.4; style_snap_residual_before_m 1.12; style_snap_residual_m 0.044; style_ref_camera_error_m 0.05; style_walls_coloured 1; style_walls_unlocked 1; style_observed_area 0.011; style_roof_observed_area 0.013; style_fill_m2 0.125; style_glb_kb_gz 2.3; style_time_s 1.1
- mdx-two-v1 · default-noise（5 次，建筑 aachen-cathedral, florence-baptistery）：jobs_failed_sfm 1; models_published 1; models_qa_rejected 1; facades_published 1; registration_rate 0.191; estimate_coverage 0.167; flagged_camera_fraction 0.0435; outline_rotation_abs_deg 45.7; outline_shift_m 8.69; outline_residual_m 0.69; outline_models 1; gps_residual_median_m 10; compass_error_median_deg 0.06; compass_photos 1; gt_cameras 4; gt_rotation_error_median_deg 0.4; gt_position_error_median_m 12.7; wall_time_s 4.3; jobs_time_s 4
- mdx-two-v1 · v0-noise（5 次，建筑 aachen-cathedral, florence-baptistery）：outline_rotation_abs_deg 26.9; outline_shift_m 4.16; outline_residual_m 0.11; gps_residual_median_m 5.2; compass_error_median_deg 0.01; gt_rotation_error_median_deg 0.586; gt_position_error_median_m 3.16; wall_time_s 12.3; jobs_time_s 13.1; style_snap_residual_before_m 0.015; style_snap_residual_m 0.001; style_ref_camera_error_m 0.02; style_observed_area 0.001; style_glb_kb_gz 0.1; style_time_s 0.7
- synthetic-v1 · v0-noise（3 次，建筑 synthetic-ring, synthetic-two-sides）：outline_rotation_abs_deg 0.025; outline_shift_m 0.005; gps_residual_median_m 0.005; gt_rotation_error_median_deg 0.011; gt_position_error_median_m 0.004; wall_time_s 6.2; jobs_time_s 6.1; style_snap_residual_before_m 0.002; style_observed_area 0.002; style_fill_m2 0.157; style_glb_kb_gz 0.3; style_time_s 0.5

## 指标定义

metric_schema_version = 3

| 指标 | 单位 | 方向 | 跨建筑汇总 | 说明 |
|---|---|---|---|---|
| photos_listed | photos | - | sum | Photos of the test set present (and verified) for this building. |
| photos_uploaded | photos | - | sum | Photos uploaded: those whose own EXIF carries GPS (all of them where the set says so). |
| photos_accepted | photos | - | sum | Photos the product's validation accepted. |
| acceptance_rate | fraction | - | mean | photos_accepted / photos_uploaded. |
| sectors_with_photos | sectors | - | sum | Sectors with at least one accepted photo. |
| sectors_face_ready | sectors | - | sum | Sectors with enough accepted photos for a facade model. |
| jobs_run | jobs | - | sum | Reconstruction jobs the planner started. |
| jobs_failed_sfm | jobs | ↓ | sum | Jobs whose reconstruction failed outright (no model at all). |
| models_published | models | ↑ | sum | Models that passed the quality gate, superseded ones included. |
| models_qa_rejected | models | ↓ | sum | Models the quality gate kept unpublished. |
| facades_published | sectors | ↑ | sum | Sectors with a published facade model at the end. |
| full_model_published | 0/1 | ↑ | sum | 1 if a full model is published at the end. |
| registration_rate | fraction | ↑ | mean | Registered / input photos over the latest attempt per target (each facade, the full model); a failed reconstruction counts as 0 registered. |
| estimate_coverage | fraction | ↑ | mean | Accepted photos for which the landmark page shows an estimated pose. |
| flagged_camera_fraction | fraction | ↓ | mean | Cameras of the latest published models withheld (disagree with their own sensors, outside the radius, degenerate focal length). |
| outline_rotation_abs_deg | deg | ↓ | median | &#124;Rotation&#124; the wall points of the latest published model per target need to fit the OSM outline (median over those models): a GPS-free check of the absolute orientation. |
| outline_shift_m | m | ↓ | median | Shift those wall points need: a GPS-free check of the absolute position (a single straight wall only constrains the part normal to it). |
| outline_residual_m | m | ↓ | median | Median wall-point distance to the outline after that fit: how well the walls match the shape. |
| outline_models | models | - | sum | Latest published models with enough wall points for the outline fit. |
| gps_residual_median_m | m | ↓ | median | Median distance of the shown camera estimates from the photos' own EXIF GPS (partly circular: GPS also drives the georegistration). |
| compass_error_median_deg | deg | ↓ | median | Median &#124;shown estimated heading - EXIF compass&#124; over the photos with a compass reading. |
| compass_photos | photos | - | sum | Shown estimates with an EXIF compass reading. |
| gt_cameras | cameras | - | sum | Registered cameras of the latest published models that have a reference pose. |
| gt_rotation_error_median_deg | deg | ↓ | median | Median camera rotation error against the reference after the best global rotation (per model, pooled over the latest published models). |
| gt_position_error_median_m | m | ↓ | median | Median camera position error after a Sim3 of the reference centres onto the model (metres of the model's frame). |
| gt_gross_cameras | cameras | ↓ | sum | Reference-posed cameras more than 45 deg or 20 m off (twin facades and the like). |
| gt_gross_shown | cameras | ↓ | sum | Of those, the ones without a flag: shown to users. |
| wall_time_s | s | ↓ | sum | Wall time of the building run. |
| jobs_time_s | s | ↓ | sum | Sum of the reconstruction jobs' run times. |
| style_models_stylized | models | - | sum | Published models (superseded ones included) the stylization ran on. |
| style_models_eligible | models | - | sum | Of those, georegistered ones (georeg_level = full): the models the registration gate is for. |
| style_registration_passed | models | ↑ | sum | Eligible models passing the registration gate (STYLIZED_MODEL 3.2: snap accepted with median wall distance <= snap_max_residual_m, and the edge-by-edge verification). |
| style_registration_pass_rate | fraction | ↑ | pooled | style_registration_passed / style_models_eligible (across buildings: pooled over all models). |
| style_registration_checked | models | - | sum | Passing models with at least 3 cameras that have an outline-corrected reference pose. |
| style_registration_wrong | models | ↓ | sum | Of those, passes that are wrong: the similarity the snapped model's cameras still need to match the outline-corrected reference exceeds 5 deg, moves its walls by more than 3 m or rescales it by more than 15% (photos would colour the wrong wall, or the wrong part of it). |
| style_registration_right | models | ↑ | sum | Of the checked passes, the right ones (the rest of style_registration_checked is wrong). |
| style_rejected_placed_right | models | ↓ | sum | Eligible models the gate refused although the georegistration alone already placed them right (at least 3 reference cameras; the correction they needed was within 5 deg, 3 m at the walls and 15%): clay where colour would have been right, the price of refusing wrong placements. |
| style_snap_refused_gps | models | - | sum | Eligible models whose snap was refused because, snapped, their cameras would stand nearer to or farther from the walls than their own EXIF GPS says (points and cameras disagree). |
| style_snap_refused_ambiguous | models | - | sum | Eligible models whose snap was refused because another placement (a parallel face, a bay away) fits about as well. |
| style_snap_residual_before_m | m | ↓ | median | Median wall-point distance to the footprint before snapping (the georegistration alone), median over the eligible models. |
| style_snap_residual_m | m | ↓ | median | The same after an accepted snap (gate condition: <= 1.5 m), median over those models. |
| style_ref_camera_error_m | m | ↓ | median | Median horizontal camera-centre error of the checked passing models after their snap, against the outline-corrected reference. |
| style_walls_total | walls | - | sum | Massing walls of the building's best stylized model (most coloured walls, then observed area). |
| style_walls_coloured | walls | ↑ | sum | Walls with photo colour in that model (unlocked + partly unlocked); the rest are clay (待解锁). |
| style_walls_unlocked | walls | ↑ | sum | Of those, unlocked (STYLIZED_MODEL 3.3: mask >= 60 % and >= 3 cameras each seeing >= 15 % of it). |
| style_observed_area | fraction | ↑ | mean | Observed wall area / all wall area of that model, observed-fill excluded. |
| style_roof_observed_area | fraction | - | mean | Observed roof area / all roof area of that model. |
| style_fill_m2 | m2 | - | sum | Small holes filled with the wall's own base colour (D12, observed-fill) in that model. |
| style_qa_failed | models | ↓ | sum | Stylized models failing the honesty QA, audit_model or audit_glb (must be 0). |
| style_provenance_violations | count | ↓ | sum | Honesty QA violations summed over all stylized models (non-clay texels outside mask and legal fill, textured or photo-claiming locked faces, faces without provenance, off-palette colours, illegal fill regions; must be 0). |
| style_glb_kb_gz | KB | ↓ | max | gzip size of the best stylized model's GLB. |
| style_time_s | s | ↓ | sum | Stylization wall time over all stylized models of the building. |
