# Crowd3D 基准排行榜（benchmark leaderboard）

由 `python -I scripts/benchmark.py report` 根据 `eval/results/<testset>/*.json` 生成，请勿手工编辑。指标定义见文末；方向：↑ 越大越好，↓ 越小越好。

## 全部运行

| 测试集 | 运行 | 配置 | 开始时间 (UTC) | 提交 | 建筑 成功/总数 | models_published (models) | models_qa_rejected (models) | facades_published (sectors) | full_model_published (0/1) | registration_rate (fraction) | estimate_coverage (fraction) | outline_rotation_abs_deg (deg) | outline_shift_m (m) | gt_rotation_error_median_deg (deg) | gt_gross_shown (cameras) | wall_time_s (s) | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mdx-two-v1 | default-noise | default | 2026-10-22 01:48 | 94ee607 | 2/2 | 4 | 1 | 4 | 0 | 0.673 | 0.558 | 26.4 | 9.84 | 0.351 | 0 | 58.5 | noise floor: the baseline five times on both buildings |
| mdx-two-v1 | fast-1 | fast | 2026-10-22 01:53 | 94ee607 | 2/2 | 3 | 1 | 3 | 0 | 0.577 | 0.475 | 1.87 | 7.85 | 0.431 | 0 | 56.2 | fast config (1024 px); these photos are 1000 px, below both caps, so the config changes nothing here: the differences to default are run-to-run variation |
| synthetic-v1 | default-1 | default | 2026-10-22 01:45 | 94ee607 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.58 | 1.16 | 0.194 | 0 | 59.8 | end-to-end validation: run 1 of 2 identical runs |
| synthetic-v1 | default-2 | default | 2026-10-22 01:46 | 94ee607 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.58 | 1.16 | 0.194 | 0 | 59.4 | end-to-end validation: run 2 of 2 identical runs |
| synthetic-v1 | fast-1 | fast | 2026-10-22 01:47 | 94ee607 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.57 | 1.17 | 0.2 | 0 | 58.4 | fast config (1024 px); these photos are 640 px, below both caps, so the config changes nothing here: the differences to default are run-to-run variation |

## 每个配置最新一次运行的分建筑结果

### mdx-two-v1 · default · default-noise

| 建筑 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | wall_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| florence-baptistery | ok | 20 | 3 | 0 | 3 | 0 | 0.821 | 0.95 | 2.59 | 7.85 | 0.33 | 7.38 | 2.51 | 0.431 | 0 | 0 | 34.4 |
| aachen-cathedral | ok | 24 | 1 | 1 | 1 | 0 | 0.524 | 0.167 | 50.1 | 11.8 | 0.81 | 7.54 | - | 0.271 | 0 | 0 | 24.1 |

### mdx-two-v1 · fast · fast-1

| 建筑 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | wall_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| florence-baptistery | ok | 20 | 3 | 0 | 3 | 0 | 0.821 | 0.95 | 1.87 | 7.85 | 0.22 | 7.38 | 2.51 | 0.431 | 0 | 0 | 35.1 |
| aachen-cathedral | ok | 24 | 0 | 1 | 0 | 0 | 0.333 | 0 | - | - | - | - | - | - | 0 | 0 | 21.1 |

### synthetic-v1 · default · default-2

| 建筑 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | wall_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic-ring | ok | 32 | 9 | 0 | 8 | 1 | 1 | 1 | 2.18 | 1.4 | 0.03 | 2.56 | 4.98 | 0.258 | 0 | 0 | 52 |
| synthetic-two-sides | ok | 12 | 3 | 0 | 2 | 0 | 1 | 0.917 | 2.98 | 0.925 | 0.025 | 2.07 | 4.41 | 0.129 | 0 | 0 | 7.4 |

### synthetic-v1 · fast · fast-1

| 建筑 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | wall_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic-ring | ok | 32 | 9 | 0 | 8 | 1 | 1 | 1 | 2.18 | 1.4 | 0.03 | 2.56 | 4.98 | 0.259 | 0 | 0 | 50.5 |
| synthetic-two-sides | ok | 12 | 3 | 0 | 2 | 0 | 1 | 0.917 | 2.96 | 0.93 | 0.025 | 2.07 | 4.41 | 0.14 | 0 | 0 | 7.9 |

## 运行间噪声（同一配置重复运行的极差）

- mdx-two-v1 · default-noise（5 次，建筑 aachen-cathedral, florence-baptistery）：jobs_failed_sfm 1; models_published 1; models_qa_rejected 1; facades_published 1; registration_rate 0.191; estimate_coverage 0.167; flagged_camera_fraction 0.0435; outline_rotation_abs_deg 45.7; outline_shift_m 8.69; outline_residual_m 0.69; outline_models 1; gps_residual_median_m 10; compass_error_median_deg 0.06; compass_photos 1; gt_cameras 4; gt_rotation_error_median_deg 0.4; gt_position_error_median_m 12.7; wall_time_s 4.3; jobs_time_s 4

## 指标定义

metric_schema_version = 1

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
