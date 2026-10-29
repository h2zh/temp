# Crowd3D 基准排行榜（benchmark leaderboard）

由 `python -I scripts/benchmark.py report` 根据 `eval/results/<testset>/*.json` 生成，请勿手工编辑。指标定义见文末；方向：↑ 越大越好，↓ 越小越好。

## 全部运行

划分：结果记录的 dev / test（`testset.split`，没记的全算 dev）；两个划分都有建筑跑过的结果分两行汇总。

| 测试集 | 运行 | 配置 | 划分 | 开始时间 (UTC) | 提交 | 建筑 成功/总数 | models_published (models) | models_qa_rejected (models) | facades_published (sectors) | full_model_published (0/1) | registration_rate (fraction) | estimate_coverage (fraction) | outline_rotation_abs_deg (deg) | outline_shift_m (m) | gt_rotation_error_median_deg (deg) | gt_gross_shown (cameras) | style_models_eligible (models) | style_registration_pass_rate (fraction) | style_registration_right (models) | style_registration_wrong (models) | style_rejected_placed_right (models) | style_walls_coloured (walls) | style_qa_failed (models) | wall_time_s (s) | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| commons-ten-v1 | v0-noise | default | dev | 2026-10-23 05:36 | e264f7e | 2/2 | 8 | 11 | 6 | 0 | 0.653 | 0.459 | 3.5 | 9.85 | 0.237 | 0 | 7 | 0.429 | - | 0 | - | 2 | 0 | 1489 | v0 noise floor on real Commons photos: 3 repeats of the 2 cheapest core buildings (metric schema 2, stylize on). Not a performance record; v0-baseline is. |
| commons-ten-v1 | v0-baseline | default | dev | 2026-10-23 05:42 | e264f7e | 5/5 | 54 | 55 | 19 | 0 | 0.662 | 0.598 | 3.82 | 10.1 | 0.744 | 7 | 51 | 0.608 | - | 19 | - | 16 | 0 | 9720 | v0 baseline on the CORE FIVE Commons buildings only (metric schema 2, style_* metrics, stylize on, keep_work_dirs). Left out of this round (the container restarts about every 3 h and a large building takes 45-50 min): sagrada-familia, hungarian-parliament, lincoln-memorial, india-gate, sydney-opera-house. Time cap: brandenburg-gate and frauenkirche-dresden hit the soft cap (no new upload after 80% of the 3600 s hard cap), so only 131/150 and 77/111 of their photos were uploaded. All five building runs are reused via --resume (same run key: code, config, test set, inputs, options); they ran 2026-10-23 01:46-03:44 UTC (four) and 04:44-05:36 UTC (frauenkirche-dresden). |
| commons-ten-v1 | v1-after-fixes | default | dev | 2026-10-24 00:25 | 1d6055b | 5/5 | 62.5 | 62 | 20 | 0 | 0.676 | 0.6 | 5.81 | 9.76 | 0.802 | 7 | 59.5 | 0 | 0 | 0 | 0 | 0 | 0 | 9613 | v1 after the round-3 registration fixes (STYLIZED_MODEL 3.2): round-2 gate (prior-only models colour nothing; deterministic grid + distance-gated ICP; ambiguity by photo displacement over the walls; turn profile; fit limit 0.45 m) plus: the cameras' displacement counts in every ambiguity check (a rival or a turn / rescale that moves the cameras >= 1.5 m, the turn profile also tries the turn that moves them 1.5 m); scale profile (+-3/6/10 % about the walls within 2 GPS scale sigmas); GT-free tilt check from the cameras' heights (<= 1.5 m at the walls, plus 1 standard error); a face carrying wall points that no camera can see past the building refuses the registration; reported yaw sigma floored at 3 deg; flagged cameras colour nothing. Metric schema 5: right = cameras' horizontal centres within 1.5 m of the reference (median) and <= 5 deg, <= 1.5 m over the wall points (90th pct, 3-D with the rotation from the cameras' orientations), <= 10 % scale. 2 repeats per building (metrics = median, noise = max-min). stylize on (its own time budget), keep_work_dirs. CORE FIVE only, as v0-baseline; left out: sagrada-familia, hungarian-parliament, lincoln-memorial, india-gate, sydney-opera-house (two repeats of the core five alone take about 5.5 h of COLMAP). |
| mdx-two-v1 | default-noise | default | dev | 2026-10-22 01:48 | 94ee607 | 2/2 | 4 | 1 | 4 | 0 | 0.673 | 0.558 | 26.4 | 9.84 | 0.351 | 0 | - | - | - | - | - | - | - | 58.5 | noise floor: the baseline five times on both buildings |
| mdx-two-v1 | fast-1 | fast | dev | 2026-10-22 01:53 | 94ee607 | 2/2 | 3 | 1 | 3 | 0 | 0.577 | 0.475 | 1.87 | 7.85 | 0.431 | 0 | - | - | - | - | - | - | - | 56.2 | fast config (1024 px); these photos are 1000 px, below both caps, so the config changes nothing here: the differences to default are run-to-run variation |
| mdx-two-v1 | v0-baseline | default | dev | 2026-10-23 01:31 | 99c006c+dirty | 2/2 | 4 | 1 | 4 | 0 | 0.673 | 0.558 | 3.52 | 14.2 | 0.551 | 0 | 2 | 0.5 | - | 0 | - | 2 | 0 | 76.8 | v0 baseline, metric schema 2 (style_* metrics), stylize on, keep_work_dirs |
| mdx-two-v1 | v0-noise | default | dev | 2026-10-23 01:34 | 99c006c+dirty | 2/2 | 4 | 1 | 4 | 0 | 0.673 | 0.558 | 12.9 | 11.8 | 0.655 | 0 | 2 | 0.5 | - | 0 | - | 2 | 0 | 86.1 | v0 noise floor (5 repeats), metric schema 2 (style_* metrics), stylize on, keep_work_dirs |
| mdx-two-v1 | v1-after-fixes | default | dev | 2026-10-24 00:19 | 3bbb0dd+dirty | 2/2 | 3.5 | 1 | 3.5 | 0 | 0.625 | 0.517 | 1.87 | 7.85 | 0.443 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 126 | v1 after the round-3 registration fixes (STYLIZED_MODEL 3.2): round-2 gate (prior-only models colour nothing; deterministic grid + distance-gated ICP; ambiguity by photo displacement over the walls; turn profile; fit limit 0.45 m) plus: the cameras' displacement counts in every ambiguity check (a rival or a turn / rescale that moves the cameras >= 1.5 m, the turn profile also tries the turn that moves them 1.5 m); scale profile (+-3/6/10 % about the walls within 2 GPS scale sigmas); GT-free tilt check from the cameras' heights (<= 1.5 m at the walls, plus 1 standard error); a face carrying wall points that no camera can see past the building refuses the registration; reported yaw sigma floored at 3 deg; flagged cameras colour nothing. Metric schema 5: right = cameras' horizontal centres within 1.5 m of the reference (median) and <= 5 deg, <= 1.5 m over the wall points (90th pct, 3-D with the rotation from the cameras' orientations), <= 10 % scale. 2 repeats per building (metrics = median, noise = max-min). stylize on (its own time budget), keep_work_dirs |
| mdx-two-v1 | v1-merged | default | dev | 2026-10-24 22:16 | 0a49b22 | 2/2 | 4 | 1 | 4 | 0 | 0.673 | 0.558 | 26.4 | 9.84 | 0.365 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 63.5 | v1-merged: the merged main (the structure-first branch with D24, merge c3416ce, plus merge-fix round 2) on the default path (no structure massing). Registration gate as v1-after-fixes (STYLIZED_MODEL 3.2, metric schema 5). Default-path changes the branch brought: colour chain (neutral white-balance anchor, per-wall WB re-anchor, stripes, roof_lift 0.2 instead of 0.5), relief_detail on (builds nothing without a structure massing and point tracks), a roof face counts as observed only when >= 1 camera faces it (roof_min_cameras 1). 2 repeats per building (metrics = median, noise = max-min). stylize on (its own time budget), keep_work_dirs |
| synthetic-v1 | default-1 | default | dev | 2026-10-22 01:45 | 94ee607 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.58 | 1.16 | 0.194 | 0 | - | - | - | - | - | - | - | 59.8 | end-to-end validation: run 1 of 2 identical runs |
| synthetic-v1 | default-2 | default | dev | 2026-10-22 01:46 | 94ee607 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.58 | 1.16 | 0.194 | 0 | - | - | - | - | - | - | - | 59.4 | end-to-end validation: run 2 of 2 identical runs |
| synthetic-v1 | fast-1 | fast | dev | 2026-10-22 01:47 | 94ee607 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.57 | 1.17 | 0.2 | 0 | - | - | - | - | - | - | - | 58.4 | fast config (1024 px); these photos are 640 px, below both caps, so the config changes nothing here: the differences to default are run-to-run variation |
| synthetic-v1 | v0-baseline | default | dev | 2026-10-23 01:25 | 99c006c+dirty | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.57 | 1.17 | 0.199 | 0 | 12 | 0.5 | - | 2 | - | 6 | 0 | 78.7 | v0 baseline, metric schema 2 (style_* metrics), stylize on, keep_work_dirs |
| synthetic-v1 | v0-noise | default | dev | 2026-10-23 01:27 | 99c006c+dirty | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.57 | 1.17 | 0.199 | 0 | 12 | 0.5 | - | 2 | - | 6 | 0 | 74.1 | v0 noise floor (3 repeats), metric schema 2 (style_* metrics), stylize on, keep_work_dirs |
| synthetic-v1 | v1-after-fixes | default | dev | 2026-10-24 00:16 | 3bbb0dd+dirty | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.57 | 1.17 | 0.199 | 0 | 12 | 0.167 | 2 | 0 | 2 | 6 | 0 | 76.7 | v1 after the round-3 registration fixes (STYLIZED_MODEL 3.2): round-2 gate (prior-only models colour nothing; deterministic grid + distance-gated ICP; ambiguity by photo displacement over the walls; turn profile; fit limit 0.45 m) plus: the cameras' displacement counts in every ambiguity check (a rival or a turn / rescale that moves the cameras >= 1.5 m, the turn profile also tries the turn that moves them 1.5 m); scale profile (+-3/6/10 % about the walls within 2 GPS scale sigmas); GT-free tilt check from the cameras' heights (<= 1.5 m at the walls, plus 1 standard error); a face carrying wall points that no camera can see past the building refuses the registration; reported yaw sigma floored at 3 deg; flagged cameras colour nothing. Metric schema 5: right = cameras' horizontal centres within 1.5 m of the reference (median) and <= 5 deg, <= 1.5 m over the wall points (90th pct, 3-D with the rotation from the cameras' orientations), <= 10 % scale. 2 repeats per building (metrics = median, noise = max-min). stylize on (its own time budget), keep_work_dirs |
| synthetic-v1 | v1-merged | default | dev | 2026-10-24 22:13 | 0a49b22 | 2/2 | 12 | 0 | 10 | 1 | 1 | 0.958 | 2.58 | 1.16 | 0.194 | 0 | 12 | 0.167 | 2 | 0 | 2 | 6 | 0 | 60.4 | v1-merged: the merged main (the structure-first branch with D24, merge c3416ce, plus merge-fix round 2) on the default path (no structure massing). Registration gate as v1-after-fixes (STYLIZED_MODEL 3.2, metric schema 5). Default-path changes the branch brought: colour chain (neutral white-balance anchor, per-wall WB re-anchor, stripes, roof_lift 0.2 instead of 0.5), relief_detail on (builds nothing without a structure massing and point tracks), a roof face counts as observed only when >= 1 camera faces it (roof_min_cameras 1). 2 repeats per building (metrics = median, noise = max-min). stylize on (its own time budget), keep_work_dirs |

## 每个配置最新一次运行的分建筑结果

### commons-ten-v1 · default · v1-after-fixes

| 建筑 | 划分 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_gate_passed_not_eligible | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_ref_camera_median_worst_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| florence-baptistery | dev | ok | 51 | 6.5 | 0.5 | 5 | 0 | 0.911 | 0.765 | 1.83 | 10.1 | 0.29 | 6.66 | 2.56 | 0.237 | 0 | 0 | 5.5 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 3.19 | - | 0 | 10 | 0 | 366 | 9.3 |
| aachen-cathedral | dev | ok | 75 | 2 | 9.5 | 1 | 0 | 0.36 | 0.173 | 6.8 | 10.6 | 0.495 | 7.64 | 2.9 | - | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26.6 | - | 0 | 30 | 0 | 447 | 3.3 |
| brandenburg-gate | dev | ok | 141 | 22 | 25 | 5 | 0 | 0.558 | 0.603 | 5.37 | 9.76 | 0.24 | 9.16 | 6.4 | 0.858 | 8.5 | 7 | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 7.63 | - | 0 | 4 | 0 | 3053 | 35 |
| leaning-tower-pisa | dev | ok | 118 | 12 | 17 | 4 | 0 | 0.721 | 0.682 | 5.81 | 7.14 | 0.2 | 8.08 | 6.12 | 1 | 1 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 3.5 | 7.03 | - | 0 | 8 | 0 | 2602 | 3.75 |
| frauenkirche-dresden | dev | ok | 83 | 20 | 10 | 5 | 0 | 0.829 | 0.777 | 12.1 | 7.75 | 0.495 | 9.4 | 10.3 | 0.746 | 1 | 0 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4.29 | - | 0 | 27 | 0 | 3146 | 38.4 |

### mdx-two-v1 · default · v1-merged

| 建筑 | 划分 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_gate_passed_not_eligible | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_ref_camera_median_worst_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| florence-baptistery | dev | ok | 20 | 3 | 0 | 3 | 0 | 0.821 | 0.95 | 2.65 | 7.85 | 0.33 | 7.38 | 2.5 | 0.46 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2.96 | - | 0 | 10 | 0 | 36.1 | 3.6 |
| aachen-cathedral | dev | ok | 24 | 1 | 1 | 1 | 0 | 0.524 | 0.167 | 50.1 | 11.8 | 0.81 | 7.54 | - | 0.271 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - | 0 | 30 | 0 | 27.4 | 0.2 |

### mdx-two-v1 · fast · fast-1

| 建筑 | 划分 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_gate_passed_not_eligible | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_ref_camera_median_worst_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| florence-baptistery | dev | ok | 20 | 3 | 0 | 3 | 0 | 0.821 | 0.95 | 1.87 | 7.85 | 0.22 | 7.38 | 2.51 | 0.431 | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | 35.1 | - |
| aachen-cathedral | dev | ok | 24 | 0 | 1 | 0 | 0 | 0.333 | 0 | - | - | - | - | - | - | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | 21.1 | - |

### synthetic-v1 · default · v1-merged

| 建筑 | 划分 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_gate_passed_not_eligible | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_ref_camera_median_worst_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic-ring | dev | ok | 32 | 9 | 0 | 8 | 1 | 1 | 1 | 2.18 | 1.4 | 0.03 | 2.56 | 4.98 | 0.259 | 0 | 0 | 9 | 1 | 1 | 0 | 1 | 0 | 1 | 4 | 0.915 | 0.12 | 4 | 4 | 0 | 52.6 | 13.1 |
| synthetic-two-sides | dev | ok | 12 | 3 | 0 | 2 | 0 | 1 | 0.917 | 2.98 | 0.925 | 0.025 | 2.07 | 4.41 | 0.129 | 0 | 0 | 3 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0.519 | 0.68 | 2 | 4 | 0 | 7.75 | 4.15 |

### synthetic-v1 · fast · fast-1

| 建筑 | 划分 | 状态 | photos_accepted | models_published | models_qa_rejected | facades_published | full_model_published | registration_rate | estimate_coverage | outline_rotation_abs_deg | outline_shift_m | outline_residual_m | gps_residual_median_m | compass_error_median_deg | gt_rotation_error_median_deg | gt_gross_cameras | gt_gross_shown | style_models_eligible | style_registration_passed | style_registration_right | style_registration_wrong | style_rejected_placed_right | style_gate_passed_not_eligible | style_snap_refused_gps | style_snap_refused_ambiguous | style_snap_residual_before_m | style_ref_camera_median_worst_m | style_walls_coloured | style_walls_total | style_qa_failed | wall_time_s | style_time_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| synthetic-ring | dev | ok | 32 | 9 | 0 | 8 | 1 | 1 | 1 | 2.18 | 1.4 | 0.03 | 2.56 | 4.98 | 0.259 | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | 50.5 | - |
| synthetic-two-sides | dev | ok | 12 | 3 | 0 | 2 | 0 | 1 | 0.917 | 2.96 | 0.93 | 0.025 | 2.07 | 4.41 | 0.14 | 0 | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - | 7.9 | - |

## 运行间噪声（同一配置重复运行的极差）

- commons-ten-v1 · v0-noise（3 次，建筑 aachen-cathedral, florence-baptistery）：registration_rate 0.0178; outline_rotation_abs_deg 2.45; outline_shift_m 1.2; outline_residual_m 0.06; gps_residual_median_m 0.2; compass_error_median_deg 0.96; wall_time_s 593; jobs_time_s 590; style_registration_passed 2; style_registration_pass_rate 0.4; style_snap_residual_before_m 1.12; style_snap_residual_m 0.044; style_ref_camera_error_m 0.05; style_walls_coloured 1; style_walls_unlocked 1; style_observed_area 0.011; style_roof_observed_area 0.013; style_fill_m2 0.125; style_glb_kb_gz 2.3; style_time_s 1.1
- commons-ten-v1 · v1-after-fixes（2 次，建筑 aachen-cathedral, brandenburg-gate, florence-baptistery, frauenkirche-dresden, leaning-tower-pisa）：jobs_failed_sfm 1; models_published 1; models_qa_rejected 1; registration_rate 0.0714; estimate_coverage 0.0392; flagged_camera_fraction 0.0191; outline_rotation_abs_deg 3.3; outline_shift_m 2.93; outline_residual_m 0.13; gps_residual_median_m 0.58; compass_error_median_deg 0.75; compass_photos 1; gt_rotation_error_median_deg 0.15; gt_position_error_median_m 0.045; gt_gross_cameras 1; gt_gross_shown 2; wall_time_s 260; jobs_time_s 260; style_models_stylized 1; style_models_eligible 1; style_snap_refused_ambiguous 1; style_snap_residual_before_m 0.894; style_glb_kb_gz 0.1; style_time_s 2.6
- mdx-two-v1 · default-noise（5 次，建筑 aachen-cathedral, florence-baptistery）：jobs_failed_sfm 1; models_published 1; models_qa_rejected 1; facades_published 1; registration_rate 0.191; estimate_coverage 0.167; flagged_camera_fraction 0.0435; outline_rotation_abs_deg 45.7; outline_shift_m 8.69; outline_residual_m 0.69; outline_models 1; gps_residual_median_m 10; compass_error_median_deg 0.06; compass_photos 1; gt_cameras 4; gt_rotation_error_median_deg 0.4; gt_position_error_median_m 12.7; wall_time_s 4.3; jobs_time_s 4
- mdx-two-v1 · v0-noise（5 次，建筑 aachen-cathedral, florence-baptistery）：outline_rotation_abs_deg 26.9; outline_shift_m 4.16; outline_residual_m 0.11; gps_residual_median_m 5.2; compass_error_median_deg 0.01; gt_rotation_error_median_deg 0.586; gt_position_error_median_m 3.16; wall_time_s 12.3; jobs_time_s 13.1; style_snap_residual_before_m 0.015; style_snap_residual_m 0.001; style_ref_camera_error_m 0.02; style_observed_area 0.001; style_glb_kb_gz 0.1; style_time_s 0.7
- mdx-two-v1 · v1-after-fixes（2 次，建筑 aachen-cathedral, florence-baptistery）：jobs_failed_sfm 1; models_published 1; facades_published 1; registration_rate 0.191; estimate_coverage 0.167; outline_models 1; compass_error_median_deg 0.01; gt_cameras 4; gt_rotation_error_median_deg 0.023; wall_time_s 22.2; jobs_time_s 23.1; style_models_stylized 1; style_snap_residual_before_m 0.015; style_glb_kb_gz 0.1; style_time_s 0.5
- mdx-two-v1 · v1-merged（2 次，建筑 aachen-cathedral, florence-baptistery）：outline_rotation_abs_deg 0.13; compass_error_median_deg 0.03; gt_rotation_error_median_deg 0.011; wall_time_s 3.6; jobs_time_s 3.4; style_snap_residual_before_m 0.02; style_time_s 0.2
- synthetic-v1 · v0-noise（3 次，建筑 synthetic-ring, synthetic-two-sides）：outline_rotation_abs_deg 0.025; outline_shift_m 0.005; gps_residual_median_m 0.005; gt_rotation_error_median_deg 0.011; gt_position_error_median_m 0.004; wall_time_s 6.2; jobs_time_s 6.1; style_snap_residual_before_m 0.002; style_observed_area 0.002; style_fill_m2 0.157; style_glb_kb_gz 0.3; style_time_s 0.5
- synthetic-v1 · v1-after-fixes（2 次，建筑 synthetic-ring, synthetic-two-sides）：gps_residual_median_m 0.005; gt_rotation_error_median_deg 0.001; gt_position_error_median_m 0.004; wall_time_s 2.4; jobs_time_s 2.4; style_ref_camera_error_m 0.01; style_ref_camera_median_worst_m 0.01; style_ref_camera_error_max_m 0.01; style_observed_area 0.002; style_fill_m2 0.11; style_glb_kb_gz 0.1; style_time_s 2.4
- synthetic-v1 · v1-merged（2 次，建筑 synthetic-ring, synthetic-two-sides）：wall_time_s 0.7; jobs_time_s 0.6; style_time_s 0.1

## 指标定义

metric_schema_version = 7

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
| cpu_time_s | s | ↓ | sum | Process CPU time (user + system, all threads) of the building run over the span of wall_time_s: the primary cost metric (wall time grows when other runs share the cores). |
| jobs_cpu_s | s | ↓ | sum | CPU time (user + system) of the reconstruction jobs. |
| peak_rss_jobs_mb | MB | ↓ | max | Highest resident memory of the process during any upload request's jobs (VmHWM, reset before the jobs; the process peak ru_maxrss where Linux does not allow the reset). |
| baseline_rss_mb | MB | - | max | Resident memory just before the jobs of that request: the part of peak_rss_jobs_mb the jobs did not add. |
| request_latency_p50_s | s | ↓ | median | Median time from the start of an upload request to the end of its jobs, over the requests that ran jobs (how long a contributor waits for the models to change). |
| request_latency_p90_s | s | ↓ | median | 90th percentile of the same. |
| photo_time_to_pose_p50_s | s | ↓ | median | Median time from a photo's upload request to the first time the landmark page shows a pose for it, over the photos that got one (estimate_coverage says how many). |
| photo_time_to_pose_p90_s | s | ↓ | median | 90th percentile of the same. |
| disk_run_mb | MB | ↓ | sum | Size of the run's data dir after the uploads (photos, models, database; the job work dirs only the harness keeps are left out). |
| style_models_stylized | models | - | sum | Published models (superseded ones included) the stylization ran on. |
| style_models_eligible | models | - | sum | Of those, georegistered ones (georeg_level = full): the models the registration gate is for. |
| style_registration_passed | models | ↑ | sum | Eligible models passing the registration gate (STYLIZED_MODEL 3.2: snap accepted with median wall distance <= snap_max_residual_m, and the edge-by-edge verification). |
| style_registration_pass_rate | fraction | ↑ | pooled | style_registration_passed / style_models_eligible (across buildings: pooled over all models). |
| style_registration_checked | models | - | sum | Passing models with at least 3 cameras that have an outline-corrected reference pose. |
| style_registration_wrong | models | ↓ | sum | Of those, passes that are wrong: the snapped model's cameras are more than 1.5 m from the outline-corrected reference (median, horizontal), or the similarity they still need exceeds 5 deg, moves the wall points by more than 1.5 m (90th percentile; in 3-D, with the rotation from the cameras' orientations, so a tilt counts) or rescales the model by more than 10% (photos would colour the wrong wall, or the wrong part of it). |
| style_registration_right | models | ↑ | sum | Of the checked passes, the right ones (the rest of style_registration_checked is wrong). |
| style_coloured_walls_wrong | walls | ↓ | sum | Walls with photo colour (unlocked + partly unlocked) summed over the passes style_registration_wrong counts: colour shown on a model placed wrong, taken from photos of another wall or another part of it (must be 0). |
| style_rejected_placed_right | models | ↓ | sum | Eligible models the gate refused although the georegistration alone already placed them right (at least 3 reference cameras; the same limits as style_registration_wrong): clay where colour would have been right, the price of refusing wrong placements. |
| style_gate_passed_not_eligible | models | ↓ | sum | Published models that are not georegistered (prior only) yet pass the registration gate: they would be coloured from a compass yaw and a guessed scale (STYLIZED_MODEL 3.2 item 4; must be 0). |
| style_snap_refused_gps | models | - | sum | Eligible models whose snap was refused because, snapped, their cameras would stand nearer to or farther from the walls than their own EXIF GPS says (points and cameras disagree). |
| style_snap_refused_ambiguous | models | - | sum | Eligible models whose snap was refused because another placement (a parallel face, a bay away) fits about as well. |
| style_snap_residual_before_m | m | ↓ | median | Median wall-point distance to the footprint before snapping (the georegistration alone), median over the eligible models. |
| style_snap_residual_m | m | ↓ | median | The same after an accepted snap (gate condition: <= 1.5 m), median over those models. |
| style_ref_camera_error_m | m | ↓ | median | Median horizontal camera-centre error of the checked passing models after their snap, against the outline-corrected reference (median over those models). |
| style_ref_camera_median_worst_m | m | ↓ | max | The largest of those per-model median camera-centre errors (a pass above 1.5 m is wrong). |
| style_ref_camera_error_max_m | m | ↓ | max | The worst single camera of the checked passing models (horizontal distance to its reference position; one camera with a bad pose of its own can be far off while the model is right). |
| style_ref_walls_p90_worst_m | m | ↓ | max | The largest 3-D wall-point displacement (90th percentile) the checked passing models still need to match the reference (2-D when the orientations are unknown). |
| style_incomplete | models | ↓ | sum | Published models the stylization did not reach (its own time cap): the building's style_* metrics then cover only the rest, and compare treats them as incomplete rather than as a change. |
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
| photo_pose_determined | photos | ↑ | sum | Uploaded photos whose position and viewing direction are both determined (sfm + loc + exif). |
| photo_pose_determined_rate | fraction | ↑ | pooled | photo_pose_determined / photos_uploaded: every rejected upload and every photo with GPS but no compass that SfM did not place counts as a failure. |
| photo_pose_sfm | photos | - | sum | Of those, posed by a camera of a published SfM model (what the landmark page shows; moved with the model's footprint snap where one was accepted). |
| photo_pose_loc | photos | - | sum | Posed by visual localization against a published model (a hook: the product has no localizer yet). |
| photo_pose_exif | photos | - | sum | Posed by the photo's own EXIF GPS and compass (approximate). |
| photo_pose_gps_only | photos | ↓ | sum | Accepted photos with a GPS position but no direction (no compass, not placed by SfM): failures. |
| photo_pose_gt | photos | - | sum | Uploaded photos with a verified reference pose (rejected ones included). |
| photo_pose_gt_determined | photos | ↑ | sum | Of those, determined. |
| photo_pose_correct | photos | ↑ | sum | Of those, within 10 m and 15 deg of the reference. |
| photo_pose_correct_rate | fraction | ↑ | pooled | photo_pose_correct / photo_pose_gt: the pass rate on the photos that can be checked (undetermined and wrong both fail). |
| photo_pose_wrong | photos | ↓ | sum | Determined photos with a reference pose more than 10 m or 15 deg off. |
| photo_pose_correct_strict | photos | ↑ | sum | Reference photos within the stricter 5 m and 10 deg. |
| photo_pose_correct_strict_rate | fraction | ↑ | pooled | photo_pose_correct_strict / photo_pose_gt. |
| photo_pose_pass | photos | ↑ | sum | Uploaded photos that pass: determined and not shown wrong (a determined photo without a reference pose cannot be checked and counts as a pass). |
| photo_pose_pass_rate | fraction | ↑ | pooled | photo_pose_pass / photos_uploaded (optimistic where references are few: quote photo_pose_correct_rate). |
| photo_pose_testset_photos | photos | - | sum | Test-set photos the protocol uploads (listed, file present, with EXIF GPS unless the set uploads every photo); more than photos_uploaded only when the harness time cap stopped the uploads. |
| photo_pose_not_uploaded | photos | ↓ | sum | photo_pose_testset_photos - photos_uploaded: photos the time cap never uploaded (failures in the *_testset rates). |
| photo_pose_determined_rate_testset | fraction | ↑ | pooled | photo_pose_determined / photo_pose_testset_photos: never-uploaded photos fail too. |
| photo_pose_gt_testset | photos | - | sum | Test-set photos with a verified reference pose, uploaded or not. |
| photo_pose_correct_rate_testset | fraction | ↑ | pooled | photo_pose_correct / photo_pose_gt_testset: the reference-subset pass rate with never-uploaded photos as failures. |
| photo_pose_pass_rate_low_testset | fraction | ↑ | pooled | photo_pose_correct / photo_pose_testset_photos: lower bound of the pass rate over every test-set photo (a photo without a reference pose counts as a failure). The reference subset over-represents photos SfM placed: quote this bound pair next to photo_pose_correct_rate. |
| photo_pose_pass_rate_high_testset | fraction | ↑ | pooled | photo_pose_pass / photo_pose_testset_photos: upper bound (a determined photo without a reference pose counts as a pass, an undetermined or never-uploaded one as a failure). |
| photo_pose_sfm_gt | photos | - | sum | SfM-posed photos with a reference pose. |
| photo_pose_sfm_correct | photos | ↑ | sum | Of those, correct. |
| photo_pose_sfm_correct_rate | fraction | ↑ | pooled | photo_pose_sfm_correct / photo_pose_sfm_gt: how far the SfM poses can be trusted. |
| photo_pose_exif_gt | photos | - | sum | EXIF-posed photos with a reference pose. |
| photo_pose_exif_correct | photos | ↑ | sum | Of those, correct. |
| photo_pose_exif_correct_rate | fraction | ↑ | pooled | photo_pose_exif_correct / photo_pose_exif_gt. |
| photo_pose_position_error_median_m | m | ↓ | median | Median horizontal position error of the determined reference photos. |
| photo_pose_heading_error_median_deg | deg | ↓ | median | Median heading error of the determined reference photos. |
| photo_pose_within_uncertainty | photos | - | sum | Determined reference photos whose errors lie inside the drawn uncertainty (sp metres and sy degrees). |
| photo_pose_uncertainty_coverage | fraction | - | pooled | photo_pose_within_uncertainty / photo_pose_gt_determined: position AND heading inside (each range is calibrated to about 85-90 % of its own errors, so both at once is less). |
| photo_pose_sfm_within_uncertainty | photos | - | sum | SfM-posed reference photos whose position and heading errors both lie inside the drawn sp / sy. |
| photo_pose_sfm_uncertainty_coverage | fraction | - | pooled | photo_pose_sfm_within_uncertainty / photo_pose_sfm_gt. |
| photo_pose_exif_within_uncertainty | photos | - | sum | EXIF-posed reference photos whose position and heading errors both lie inside the drawn sp / sy. |
| photo_pose_exif_uncertainty_coverage | fraction | - | pooled | photo_pose_exif_within_uncertainty / photo_pose_exif_gt. |
| geo_models_checked | models | - | sum | Published georegistered models with at least 3 unflagged cameras that have a verified reference pose (superseded ones included). |
| geo_camera_error_median_m | m | ↓ | median | Horizontal distance of a checked model's cameras from their reference positions (median per model, then over the models): the geographic error of the poses the page shows. |
| geo_yaw_error_median_deg | deg | ↓ | median | &#124;Yaw&#124; of the 3-D similarity moving a checked model onto the reference, median over the models: the anchor's heading error. |
| geo_scale_error_median_pct | % | ↓ | median | &#124;scale - 1&#124; of that similarity, in per cent: the anchor's scale error. |
| geo_tilt_error_median_deg | deg | ↓ | median | Tilt of that similarity: how far the model's vertical is off. |
| geo_wall_error_p90_median_m | m | ↓ | median | 90th percentile of how far that similarity moves the model's wall points (2-20 m up, in 3-D), median over the models: the geographic error at the walls. |
| wrong_face_models_published | models | ↓ | sum | Checked models whose yaw is 30 deg or more off: placed facing the wrong way (a symmetric alias, a failed yaw). |
| wrong_face_cameras_shown | cameras | ↓ | sum | Poses the landmark page shows at the end (one per photo, crowd3d.api.camera_estimates) for photos with a verified reference pose that look 30 deg or more off its heading or stand 45 deg or more round the building from it (bearing from the landmark centre). |
| cov_claimed_m | m | - | sum | Outline metres on which the latest published models show walls: at least 10 published wall points land on the bin (within 5 m). Needs only the outline. |
| cov_unverified_m | m | - | sum | Of those, the metres only models without a similarity to the reference claim (fewer than 3 oriented reference cameras): not checked. |
| cov_local_supported_m_1p5 | m | ↑ | sum | Outline metres where a model has at least 10 wall points that, moved by its similarity to the reference, lie within 1.5 m of the outline: supported in the building-local frame. The primary coverage metric. |
| cov_local_supported_m_5 | m | ↑ | sum | The same within 5 m: every covered metre (points farther than 5 m are not on the outline), the perimeter the checked models cover. |
| cov_geo_supported_m_1p5 | m | ↑ | sum | Outline metres where a model has at least 10 published wall points within 1.5 m (3-D) of where they are on the outline: supported geographically (the georegistration error included; expected near 0 today). |
| cov_geo_supported_m_5 | m | ↑ | sum | The same within 5 m. |
| cov_local_wrong_m_1p5 | m | ↓ | sum | Covered metres no model supports within 1.5 m in the building-local frame (cov_local_supported_m_5 - cov_local_supported_m_1p5): the walls' shape is off there. |
| cov_geo_wrong_m_5 | m | ↓ | sum | Covered metres no model supports geographically within 5 m (cov_local_supported_m_5 - cov_geo_supported_m_5). |
| cov_local_majority_m_1p5 | m | ↑ | sum | Outline metres where a model's points within 1.5 m in the building-local frame are at least 10 and at least 50% of its moved points landing on the bin: the share-based companion of cov_local_supported_m_1p5, which counts points and so rises with point density (10 of 100 points on the wall support a bin); this one does not rise with density alone, and drops bins seen mostly behind a colonnade. |
| cov_local_wrong_sum_m_1p5 | m | ↓ | sum | Summed over the shown models: the metres each covers but does not support within 1.5 m in the building-local frame. cov_local_wrong_m_1p5 counts a bin once, and not at all when another shown model supports it; this counts every shown model's wrong walls. |
| cov_geo_wrong_sum_m_5 | m | ↓ | sum | The same geographically within 5 m: summed over the shown models, the metres each covers but does not support (next to cov_geo_wrong_m_5). |
| calib_yaw_n | models | - | sum | Checked models (geo_models_checked, with the 3-D similarity) with a yaw sigma: z = &#124;yaw&#124; (geo_yaw_error) / sqrt(sigma_model^2 + sigma_ref^2), sigma_model = the anchor's yaw sigma (a model JSON without an anchor: yaw_sigma_deg), sigma_ref = 1 deg (0 for generator truth). |
| calib_yaw_within2 | models | - | sum | Of those, &#124;z&#124; <= 2. |
| calib_yaw_cover2 | fraction | - | pooled | calib_yaw_within2 / calib_yaw_n: the share inside 2 sigma (0.954 when the sigma is right; fewer than 20 models say little). |
| calib_yaw_beyond3 | models | ↓ | sum | Of those, &#124;z&#124; > 3: the georegistration claimed far more certainty than it had. |
| calib_scale_n | models | - | sum | Checked models (geo_models_checked, with the 3-D similarity) with a scale sigma: z = &#124;ln scale&#124; (geo_scale_error) / sqrt(sigma_model^2 + sigma_ref^2), sigma_model = the anchor's log-scale sigma (without an anchor: scale_sigma), sigma_ref = 2 % (0 for generator truth). |
| calib_scale_within2 | models | - | sum | Of those, &#124;z&#124; <= 2. |
| calib_scale_cover2 | fraction | - | pooled | calib_scale_within2 / calib_scale_n: the share inside 2 sigma (0.954 when the sigma is right; fewer than 20 models say little). |
| calib_scale_beyond3 | models | ↓ | sum | Of those, &#124;z&#124; > 3: the georegistration claimed far more certainty than it had. |
| calib_xy_n | models | - | sum | Checked models (geo_models_checked, with the 3-D similarity) with a xy sigma: z = the horizontal error at the anchor's pivot (east, north: the 2-D radius, calib_z) / sqrt(sigma_model^2 + sigma_ref^2), sigma_model = the anchor's east/north sigma (none without an anchor), sigma_ref = the reference's median wall distance, at least 0.5 m per axis (0 for generator truth). None when no checked model has an anchor (rescore --regeo replays one). |
| calib_xy_within2 | models | - | sum | Of those, &#124;z&#124; <= 2. |
| calib_xy_cover2 | fraction | - | pooled | calib_xy_within2 / calib_xy_n: the share inside 2 sigma (0.954 when the sigma is right; fewer than 20 models say little). |
| calib_xy_beyond3 | models | ↓ | sum | Of those, &#124;z&#124; > 3: the georegistration claimed far more certainty than it had. |
| calib_tilt_n | models | - | sum | Checked models (geo_models_checked, with the 3-D similarity) with a tilt sigma: z = the tilt about east and north (the 2-D radius, calib_z) / sqrt(sigma_model^2 + sigma_ref^2), sigma_model = the anchor's tilt sigma per axis (none without an anchor), sigma_ref = 0.5 deg per axis (0 for generator truth). None when no checked model has an anchor (rescore --regeo replays one). |
| calib_tilt_within2 | models | - | sum | Of those, &#124;z&#124; <= 2. |
| calib_tilt_cover2 | fraction | - | pooled | calib_tilt_within2 / calib_tilt_n: the share inside 2 sigma (0.954 when the sigma is right; fewer than 20 models say little). |
| calib_tilt_beyond3 | models | ↓ | sum | Of those, &#124;z&#124; > 3: the georegistration claimed far more certainty than it had. |
| calib_sigma_missing | models | ↓ | sum | Checked models (with the 3-D similarity) without an anchor whose georegistration left no yaw or no scale sigma (the snap then has no window to search; an anchor always has every sigma). |
| stale_overwrites | models | ↓ | sum | Published models that replaced their target's previous published model although it had photos they lack, uploaded later than any of theirs (an older input set shown over a newer one; must be 0). |
| version_drift_median_m | m | ↓ | median | Per pair of consecutive published versions of a target, the median 3-D move of their common unflagged cameras; median over the pairs. |
| version_drift_p90_m | m | ↓ | median | 90th percentile of the same over the pairs. |
| struct_models | models | - | sum | Stylized models built from an OSM parts snapshot. |
| struct_d15_adopted | count | - | sum | Active D15 adoptions (photo-measured heights). |
| struct_d25_records | count | - | sum | D25 records (photo-measured frame corrections). |
| struct_photo_share | fraction | - | mean | Share of the structure's numbers measured from photos (crowd3d.structmetrics). |
| struct_checked | count | - | sum | Adopted heights, lower bounds and tilts checked against the reference points. |
| struct_incorrect | count | ↓ | sum | Of those, outside the D15 agreement tolerance of the reference (must be 0). |
| struct_unbacked | count | ↓ | sum | Structure values without the photo evidence their record names (must be 0). |
| struct_adopted_unregistered | count | ↓ | sum | Active D15 adoptions in models the registration gate refused (must be 0). |
| struct_qa_failed | models | ↓ | sum | Models failing the structure's QA (must be 0). |
| struct_time_s | s | ↓ | sum | Time of the structure step over the building's models. |
| struct_peak_rss_mb | MB | ↓ | max | Peak resident memory of the structure step. |
