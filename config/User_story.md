# User guide for running experiments

This guide describes `config/batch_feasibility_config.yaml`: robots, input/output paths, solver choice, EAIK branch weights, waypoint densification, singularity and continuity checks, and CSV-related toggles. The same `FeasibilityConfig` is passed into `feasibility_analysis.process_toolpath()` for each toolpath CSV (implementation: `utils/feasibility/pipeline_runner.run_feasibility_pipeline`)—edit the YAML to change behaviour without changing code.

---

## EAIK multi-solution weights (`eaik_multi_solution.weights`)

Only applies when `solver: "eaik"` and `eaik_multi_solution.enabled: true`. For each waypoint, every valid IK branch is scored; the lowest **total** cost wins.


| Key                      | Effect                                                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **c0**                   | Weight on joint-space distance to the previous selected configuration (larger → stronger preference to stay on the same branch / avoid jumps).        |
| **singularity**          | Weight on the wrist-singularity penalty (J5 band) or soft σ penalty depending on internal flags — higher → more avoidance of singular configurations. |
| **manipulability**       | Weight on manipulability **reward** (subtracted from cost). `0.0` disables it.                                                                        |
| **branch_discontinuity** | One-time penalty each time the mixed-branch selector **switches** cfx along the path — higher → fewer allowed branch changes.                         |


Increasing a weight makes that term dominate the sum when choosing branches; decreasing it relaxes that criterion.

---

## Waypoint density & interpolation (`waypoint_density`)

Runs **before** IK on the Cartesian toolpath. It measures Euclidean segment lengths in mm and decides if segments are too sparse.

- `**check_frequency_hz`** — Assumes you want at most one “check” per `1/frequency` seconds along the path at the **segment TCP speed**. The allowed spacing per segment is `speed / check_frequency_hz` (capped by `max_gap_mm` below). **Higher Hz → tighter allowed spacing → more segments flagged sparse → more interpolation** (when `interpolate_sparse` is on).
- `**max_gap_mm`** — Hard cap on allowed gap (mm) between consecutive waypoints. **Lower** → stricter → more sparse segments detected → denser path. **Higher** → allows longer jumps without inserting points.
- `**interpolate_sparse`** — If `true`, sparse segments get extra poses (linear XYZ + quaternion SLERP). If `false`, density is only reported; the path stays as in the CSV.
- `**default_speed_mm_s**` — Used for segments when per-segment speeds are missing or shorter than the number of segments (feeds the density limit calculation together with `check_frequency_hz`).

**Rule of thumb:** To **increase** sampling density along the path, raise `check_frequency_hz`, lower `max_gap_mm`, and keep `interpolate_sparse: true`. To **reduce** inserted points, lower Hz, raise `max_gap_mm`, or turn interpolation off.

**Position vs orientation:** Sparsity is decided from **translation only** (Euclidean distance between consecutive XYZ waypoints). When a segment is subdivided, **the same** scalar parameter `t` along the segment is used for both: **linear** interpolation of position and **SLERP** of the quaternion. There is **no** separate YAML control for “denser rotation than translation” or different max gaps for orientation vs position—one subdivision count applies to both.

---

## Singularity (`singularity`)

- `**enabled`** — If `false`, singularity-related plots and thresholding in the pipeline are skipped where gated by this flag (reachability/IK still run).
- `**mode**` — `**unified**`: one check based on Jacobian minimum singular value vs `threshold` (fast). `**classified**`: shoulder / elbow / wrist–style breakdown using `type_thresholds` (more detail, heavier).
- `**threshold**` — In unified mode, σ_min below this flags “near singular” (see plots).
- `**type_thresholds**` — Used in **classified** mode only (`wrist`, `shoulder`, `elbow`) to tune per-type sensitivity.
- `**check_j5_only`** — Passed into **singularity analysis** (`SingularityAnalyzer` in `core/checks/singularity.py`): fast J5-based wrist check vs wrist sub-Jacobian when `false`. **EAIK branch scoring** uses `score_ik_solution_breakdown` in `core/feasibility/eaik_scoring.py` (re-exported from `core/feasibility_checks`), which follows a fixed J5-band rule (`USE_J5_SINGULARITY_ONLY`); changing this YAML alone does not toggle the σ_min vs J5 term in that scorer unless code is updated. The J5 band geometry is `j5_wrist_singularity_band_active` in the same `singularity` module.
- `**j5_threshold_deg`** — Band for wrist-near-singularity (degrees); used for J5 binary plots, `score_ik_solution_breakdown` when using the J5 term, and related checks.
- `**generate_graphs**` — Turns singularity-related PNGs on or off.

---

## Continuity (`continuity`)

Covers **C0** (position-level joint jumps) and **C1** (velocity/acceleration vs limits after TOPP-RA).

- `**enabled`** — If `false`, C1 check is skipped and continuity graphs controlled by this group are not produced.
- `**pose_scale_m_per_rad**` — Reserved for unified pose–joint metrics in supporting utilities; **not** used by the main C0/C1 plots produced by the feasibility pipeline (`utils/feasibility/plotting_trajectory.py` / aggregated plots—those use joint distances and TOPP-RA output directly). Safe to leave at default unless you extend tooling that reads it.
- `**safety_factor`** — Multiplier on joint velocity and acceleration **limits** when `check_c1_continuity` decides pass/fail (e.g. `1.05` → allow 5% headroom before flagging violation).
- `**default_speed_mm_s`** — Default TCP speed (mm/s) passed from `**feasibility_analysis_batch.py**` into `process_toolpath()` when the batch runs, so analyses use this as the nominal speed unless the CSV supplies speeds.
- `**generate_graphs**` — C0/C1 and related continuity figures.

---

## Miscellaneous — CSV loading and RobotStudio comparison

### Toolpath input: `FILTER_ONLY_IS_AT_WAYPOINT` (`utils/csv_loader_toolpath.py`)

If you set `**FILTER_ONLY_IS_AT_WAYPOINT = True**` in code, `**load_toolpath_trajectories_ext()**` keeps **only** rows where the column `**is_at_waypoint`** equals **1**. That shrinks the **main** toolpath (waypoints fed to IK / TOPP) to “arrival” samples.

**Expected CSV:** Header row with toolpath pose columns (`x,y,z,qw,qx,qy,qz` or aliases such as `rs_x_mm`, …) and optionally `**is_at_waypoint`** (0/1). Rows are numeric; first column is not treated as a header if it parses as a float.

This flag affects **trajectory loading for analysis**, not RobotStudio reference loading (below).

### RobotStudio reference: `filter_is_At_Waypoint_Rs_data` (`utils/csv_loader_toolpath.py`)

`**load_robotstudio_reference()`** optionally loads `**rs_j1_deg` … `rs_j6_deg**` and `**rs_x_mm` … `rs_qz**` for overlays and comparison plots.

If `**filter_is_At_Waypoint_Rs_data = True**` and the CSV has `**is_at_waypoint**`, **only rows with value 1** are kept for **RobotStudio** joints/TCP. The **toolpath** CSV used for IK is unchanged; only the **reference** arrays used for comparison (e.g. EAIK vs RS plots, RS cfx scoring via `utils/feasibility/robotstudio_overlay.py`) are filtered.

**Typical RobotStudio export columns:** `time_ms`, speeds/accel, `rs_x_mm`, `rs_y_mm`, `rs_z_mm`, `rs_qw`–`rs_qz`, `cf1`, `cf4`, `cf6`, `cfx`, `rs_j1_deg`–`rs_j6_deg`, `is_at_waypoint`. Joints + TCP should be aligned row-wise; TCP xyz is used for `compute_ecfx(..., target_position=...)` when present.

---

## Other YAML blocks (short)

- `**robots_to_use` / `knife_poses_to_use` / `toolpaths_folder` / `output_folder` / `use_base_frame`** — Batch I/O and frame handling.
- `**solver**` — `"eaik"` or `"pin"`.
- `**output.level1_only**` — Skip higher-level scoring outputs when true.
- `**max_ik_failures_per_trajectory**` — Stop early after N IK failures on a trajectory.
- `**reachability` / `manipulability` / `topp_ra**` — `enabled` (where applicable) + `generate_graphs` per group.
- `**ranking**` — Used by **combinatorial search**, not the batch feasibility script.

---

## Custom config file

`feasibility_analysis_batch.py` loads `**config/batch_feasibility_config.yaml` by default**. To run with another YAML (different robots, folders, weights, etc.), pass it explicitly:

```bash
python feasibility_analysis_batch.py --config path/to/your_experiment_config.yaml
```

Short form: `-c` instead of `--config`. Optional: `-o` / `--output` to override the output directory from the file.