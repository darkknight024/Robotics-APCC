# Feature 4 — Collision checking (`core.collision`)

Collision checking: discrete configuration tests at joint vector **q** (degrees), integrated with EAIK feasibility and optional RobotStudio validation. Geometry uses **Pinocchio** collision models backed by **coal** / **hpp-fcl** (broad-phase + narrow-phase on triangle meshes and convex pieces).

---

## 1. Developer story

### 1.1 Design intent

We need a single, repeatable answer to: *“Is configuration **q** in collision?”* — for robot self-contact, static cell obstacles, and (in tests) artificial forbidden regions in joint space. That answer plugs into Feature 2 as a **CFX-branch gate**: EAIK enumerates joint solutions, then colliding slots are excluded from mixed-branch selection so the emitted waypoint uses a collision-free branch when one exists.

The production path is **task-space geometry** (URDF collision meshes + fixed environment STLs). A separate **C-space** gate exists for deterministic offline tests without loading meshes or depending on FK accuracy.

**Scope today:** collision at **discrete** configurations (waypoints / IK samples). Continuous motion between waypoints is **not** checked unless you explicitly use `TrajectoryCollisionChecker` on a dense `q` sequence.

**Known limitation:** URDF collision meshes are coarser than ABB RobotStudio’s internal model. Calibration removes pairs that collide in *every* random sample (typical mesh overlap at joint interfaces), but agreement with RS is empirical, not guaranteed.

### 1.2 Algorithm

For each query configuration **q**:

1. **Forward placement** — Pinocchio places all collision bodies (robot links from URDF; environment meshes fixed in the robot base frame via `SE3`).
2. **Active pairs** — Only pre-registered pairs are tested (self: all non-adjacent link pairs; scene: robot–environment pairs plus self; whitelisted pairs are **removed** from the pair list).
3. **Margins** — Per-geometry `collision_tolerance_m` from YAML is applied as FCL `security_margin` on the pair (inflation before contact).
4. **Detection** — `pin.computeCollisions(..., stop_at_first_collision=True)` for fast boolean queries; full `check()` also runs `computeDistances` for clearance diagnostics.
5. **Calibration (optional)** — Before use, sample `q` at neutral + uniform random in joint limits; any pair that collides in **all** samples is dropped (reduces structural false positives from overlapping STL shells).

C-space checking bypasses geometry: **q** is in collision if it lies inside any configured axis-aligned box in joint space (all listed joints must fall within their bands).

### 1.3 Module map

| Location | Class / symbol | Responsibility |
|----------|----------------|----------------|
| `types.py` | `CollisionResult`, `WaypointCollisionResult`, `TrajectoryCollisionReport` | Structured outputs for `check()` and trajectory sweeps |
| `geometry.py` | `build_robot_collision_geometry`, `append_fixed_scene_geometry`, `se3_from_collision_object_pose`, `pad_q`, `ensure_collision_requests` | Load URDF collision geometry; attach static meshes; pose contract (env **xyz in mm**, quat **[qw,qx,qy,qz]**); pad **q** to full `model.nq` |
| `scene_config.py` | `CollisionObjectSpec`, `CollisionObjectsFile` | Parse `config/collision_objects.yaml` (objects, whitelists, midsole/knife names) |
| `mesh_processing.py` | `effective_mesh_path` | Prefer simplified/cached STL; optional trimesh decimation |
| `pair_rules.py` | `add_robot_self_pairs`, `remove_adjacent_pairs`, `add_robot_environment_pairs`, `index_pair_for_names` | Build and filter `GeometryModel.collisionPairs` |
| `self_checker.py` | `SelfCollisionChecker` | URDF-only self-collision; `calibrate()`; `has_collision` / `check` |
| `scene_checker.py` | `SceneCollisionChecker` | Self + static env; `from_urdf_and_scene_yaml()`; `has_collision`, `has_self_collision`, `has_environment_collision` |
| `object_checker.py` | `ObjectCollisionChecker` | Thin wrapper: robot vs environment subset of scene |
| `midsole_checker.py` | `MidsoleCollisionChecker` | Policy/diagnostics around midsole–tool whitelisted contact |
| `trajectory_checker.py` | `TrajectoryCollisionChecker` | Sweep `has_collision` over a list of **q**; first-hit reporting |
| `cspace_config.py` | `ForbiddenZoneDeg`, `JointBandDeg`, `load_cspace_forbidden_zones` | Load/test C-space YAML; validate bands against `robots_config.yaml` joint limits |
| `cspace_checker.py` | `CSpaceForbiddenChecker` | Pure joint-space forbidden hyper-rectangles; `has_collision(q)` |
| `factory.py` | `build_collision_checker_for_feasibility`, `CompositeCollisionChecker` | Assemble scene and/or C-space gates for the feasibility pipeline |
| `__init__.py` | Public exports | Stable import surface for the rest of the repo |

**Legacy shim:** `core/collision_checker.py` re-exports `SelfCollisionChecker` for older scripts.

### 1.4 Integration outside this package

Collision is not implemented inside `core/collision` alone; feasibility owns orchestration:

| Location | Role |
|----------|------|
| `core/feasibility/collision_gate.py` | Shared predicates: `is_cfx_slot_collision_free`, `first_collision_free_cfx_q`, `annotate_cfx_collision_blocked` |
| `core/feasibility/analyzer.py` | Annotates per-CFX collisions; mixed-branch selection skips colliding slots; selected path must be collision-free |
| `core/feasibility/cfx_branch_selection.py` | Mixed/global **cfx** scoring skips colliding branches before coverage/cost |
| `core/eaik_ik_solver.py` | Enumerates all CFX slots (joint limits + FK). Feature 2 does not attach the collision checker here. |
| `utils/feasibility/pipeline_runner.py` | Builds checker from config + CLI; attaches it to the analyzer only |
| `utils/config_loader.py` | `CollisionConfig` dataclass |
| `feasibility_analysis.py` | CLI + `CollisionRunOverrides`; `--no-collision` disables the Feature 2 gate |

**Contract for any gate:** implement `has_collision(q: np.ndarray) -> bool` (`True` = infeasible). Optional `check(q) -> CollisionResult` for diagnostics.

### 1.5 Configuration files

| File | Purpose |
|------|---------|
| `config/collision_objects.yaml` | Production obstacles (STL paths, poses, `mesh_scale`, `collision_tolerance_m`, per-object `whitelist_pairs`) |
| `config/collision_objects_empty.yaml` | Robot self pairs only (no env meshes) — used when validating RS trajectories recorded without test obstacles |
| `config/cspace_forbidden_zones_irb1300_714.yaml` | Five forbidden joint-space zones for IRB 1300-7/1.4 (offline tests) |
| `config/robots_config.yaml` | `joint_limits_deg` per robot (validates C-space bands) |
| `config/batch_feasibility_config.yaml` | `collision:` block — **enabled by default** for feasibility runs |

### 1.6 Extending the stack

- **New static obstacle:** add an entry under `collision_objects.yaml`; ensure `mesh_path` and `pose` match the cell frame (mm + quaternion). Rebuild is automatic on next `SceneCollisionChecker.from_urdf_and_scene_yaml`.
- **Allowed contact (e.g. knife on midsole):** add the geometry names to `whitelist_pairs` on the object, or set `midsole_geom_name` / `knife_blade_geom_name` in the YAML root.
- **New checker in feasibility:** pass any object with `has_collision` into `FeasibilityAnalyzer` and `create_solvers`, or extend `build_collision_checker_for_feasibility`.
- **Dense path validation:** wrap your checker in `TrajectoryCollisionChecker(collision_fn=checker.has_collision)`.

---

## 2. User story

### 2.1 What you get when collision is on

With collision enabled, the pipeline aims to produce **collision-free EAIK branches** per waypoint:

1. EAIK drops any IK solution in collision before branch selection.
2. If **all eight** **cfx** slots collide, the waypoint is **unreachable** (`collision_all_branches`).
3. Mixed **cfx** selection (Feature 2) never scores a branch that fails `has_collision`.
4. Final `joint_angles_rad` are audited; `collision_output_leak_count` should be **0**.

Trajectory **level-1 pass** also requires `collision_ok` when checking is active (`reachability` + `C0` + `C1` + collision).

### 2.2 Enabling / disabling in the feasibility pipeline

**Default:** collision is **ON** in `config/batch_feasibility_config.yaml`:

```yaml
collision:
  enabled: true
  scene_yaml: config/collision_objects.yaml
  scene_calibrate: true
  scene_calibrate_n_samples: 10
  scene_calibrate_seed: 42
  cspace_forbidden_yaml: null   # production: leave null
  cspace_only: false
```

**Single toolpath (Feature 2):**

```bash
python feasibility_analysis.py -t path/to/toolpath.csv -u path/to/robot.urdf
```

Collision is active unless you pass `--no-collision`.

**Batch:**

```bash
python feasibility_analysis_batch.py
python feasibility_analysis_batch.py --workers 4
```

Batch jobs call the same path as the single-toolpath CLI via `feasibility_analysis.run_batch_single_job`.

**CLI overrides:**

| Flag | Effect |
|------|--------|
| *(none)* | Scene collision ON (`collision_objects.yaml` + URDF self) |
| `--no-collision` | No checker; IK/feasibility ignore collision |
| `--cspace-forbidden-yaml PATH` | Add C-space zones (see below) |
| `--cspace-only` | With C-space YAML: **no meshes** — joint-space boxes only (offline / RS replay tests) |

**Offline C-space test example** (pre-recorded joint CSVs, no environment meshes):

```bash
python feasibility_analysis.py -t ... -u ... \
  --cspace-forbidden-yaml config/cspace_forbidden_zones_irb1300_714.yaml \
  --cspace-only
```

**Programmatic (library):**

```python
from utils.config_loader import load_batch_config
from feasibility_analysis import process_toolpath, CollisionRunOverrides

cfg = load_batch_config("config/batch_feasibility_config.yaml")
result = process_toolpath(
    toolpath_path, urdf_path, cfg,
    collision_overrides=CollisionRunOverrides(no_collision=False),
)
```

Use `CollisionRunOverrides(cspace_forbidden_yaml="...", cspace_only=True)` for C-space-only tests.

**EAIK multi-branch:** ensure `solver: eaik` and `eaik_multi_solution.enabled: true` in the batch config (default in project configs). Collision gating is most meaningful with all **cfx** branches available for selection.

### 2.3 Stand-alone collision checking

**Full scene (self + obstacles):**

```python
from core.collision import SceneCollisionChecker

checker = SceneCollisionChecker.from_urdf_and_scene_yaml(
    urdf_path="Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF.urdf",
    scene_yaml_path="config/collision_objects.yaml",
    calibrate=True,
    fixture_name="ee_link",
)
q = ...  # (6,) or (n_joints,) radians
if checker.has_collision(q):
    detail = checker.check(q)
    print(detail.colliding_pairs, detail.min_distance_m)
```

**Self-collision only:**

```python
from core.collision import SelfCollisionChecker

checker = SelfCollisionChecker(urdf_path="...")
checker.calibrate(n_samples=10, seed=42)
assert not checker.has_collision(q_neutral)
```

**C-space forbidden zones (tests):**

```python
from core.collision import CSpaceForbiddenChecker

gate = CSpaceForbiddenChecker.from_yaml("config/cspace_forbidden_zones_irb1300_714.yaml")
gate.has_collision(q_rad)
gate.colliding_zone_names(q_rad)
```

**Feasibility factory (same as pipeline):**

```python
from core.collision import build_collision_checker_for_feasibility

checker = build_collision_checker_for_feasibility(
    urdf_path=urdf,
    scene_yaml="config/collision_objects.yaml",
    cspace_forbidden_yaml=None,
)
```

**Trajectory sweep:**

```python
from core.collision import TrajectoryCollisionChecker

traj_chk = TrajectoryCollisionChecker(checker.has_collision)
report = traj_chk.check_path(list_of_q_arrays)
```

### 2.4 Interpreting feasibility outputs

Per trajectory, inspect:

- `feasibility_flags.collision_check_enabled` — was a checker attached?
- `feasibility_flags.collision_ok` — no collision rejects and no output leaks?
- `collision_reject_count` — waypoints failed for collision (single-branch or all branches)
- `collision_output_leak_count` — should be **0**; non-zero indicates a logic bug or bypass

`analysis_report.txt` includes a **Collision: PASS/FAIL** line when checking was enabled.

---

## 3. Testing and validation

### 3.1 Unit tests (C-space, no Pinocchio scene required)

```bash
python -m unittest tests.test_cspace_forbidden -q
```

Covers YAML load, band validation against `robots_config.yaml`, hit/miss queries, and rejection of single-point bands.

**C-space validation script** (writes timestamped JSON under Internal_Collision Results):

```bash
python Robot_APCC/Experiments/Internal_Collision/run_cspace_gating_validation.py
```

### 3.2 `tests/test_collision.py` — validation harness

This script does **not** implement geometry; it loads experiment YAML, runs checkers on external datasets, and writes `summary.json` / `summary.md`.

**CLI:**

| Argument | Description |
|----------|-------------|
| `--smoke` | Minimal import test: build `SelfCollisionChecker` + `SceneCollisionChecker` on default URDF (no RS data). Default if no other mode is given. |
| `--config PATH` | Run experiments defined in YAML (see `tests/collision_validation_example.yaml`). |
| `--internal-collision` | Shorthand for `--config tests/configs/internal_collision_validation.yaml`. |
| `--out DIR` | Output directory. Default: `Robot_APCC/Experiments/Internal_Collision/Results/MM_DD_YY_HH_MM_SS` for internal collision; else `tests/collision_validation_results/<YYYYMMDD_HHMMSS>`. |

**Examples:**

```bash
# Quick sanity (no dataset)
python tests/test_collision.py --smoke

# Internal_Collision RS trajectory suite (shipped CSVs under Robot_APCC/Experiments/Internal_Collision/csv/)
python tests/test_collision.py --internal-collision

# Custom validation config
python tests/test_collision.py --config tests/configs/my_validation.yaml --out /tmp/collision_out
```

**Internal collision experiment** (`internal_collision_rs_trajectories`):

- **Non-collision** CSVs (`non_collision_traj_*.csv`): checked with `scene_yaml_no_env` (`collision_objects_empty.yaml`) — expect **zero** hits.
- **Collision-expected** CSVs (`collision_traj_*.csv`, `traj*_obstacle*.csv`): checked with full `collision_objects.yaml` — expect **≥1** hit per file.
- Rows: by default `waypoint_rows_only: true` (`is_at_waypoint == 1`) to avoid double-counting RS motion/dwell duplicate rows.
- Metrics: `n_hits` = rows where **our** `has_collision(q)` is true (not RS `is_collision` column).

**Other experiment types** (require you to supply RobotStudio-labelled CSVs — see example YAML):

| `type` | Purpose |
|--------|---------|
| `joint_table_binary` | Precision/recall vs `rs_collision` column (`checker: self` or `scene`) |
| `waypoint_table_binary` | Per-waypoint binary labels |
| `tolerance_series` | Monotone response vs `collision_tolerance_m` |
| `decimation_series` | Mesh decimation sensitivity |

Copy `tests/collision_validation_example.yaml` to `tests/configs/collision_validation.yaml` and set `rs_csv` paths.

### 3.3 Feasibility + collision (end-to-end)

**Production-style** (meshes + EAIK branching):

```bash
python feasibility_analysis.py \
  -t Robot_APCC/Experiments/Internal_Collision/csv/non_collision_traj_1.csv \
  -u "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF.urdf" \
  -c config/batch_feasibility_config.yaml \
  --base_frame
```

Expect collision **on** in the console (`Collision: ON (scene=...)`). Use `--no-collision` to compare behaviour without gating.

**C-space-only regression** (deterministic, no meshes):

```bash
python feasibility_analysis.py -t ... -u ... \
  --cspace-forbidden-yaml config/cspace_forbidden_zones_irb1300_714.yaml \
  --cspace-only
```

### 3.4 Validation performed in this repo

| Activity | Data | Pass criteria |
|----------|------|----------------|
| Internal RS trajectories | `Robot_APCC/Experiments/Internal_Collision/csv/` | Non-collision files: `n_hits == 0` with empty env; collision files: `n_hits > 0` with full scene |
| C-space gating | Synthetic **q** in `run_cspace_gating_validation.py` | Known inside/outside boxes per zone |
| Unit tests | `tests/test_cspace_forbidden.py` | Load, limits, hit/miss |

**Replicate internal collision validation:**

```bash
cd /path/to/Robotics-APCC
python tests/test_collision.py --internal-collision
# Inspect: Robot_APCC/Experiments/Internal_Collision/Results/<timestamp>/summary.md
```

Ensure Pinocchio/coal dependencies are installed and URDF/STL paths resolve from the repo root.

**Note:** Our scene checker may report more waypoint hits than RS `is_collision` on some files — URDF calibration and obstacle placement differ from RobotStudio. The harness tests **separation** (clear vs obstructed paths), not bit-exact agreement with RS labels.

### 3.5 Dependencies

- **Pinocchio** with collision support (coal / hpp-fcl)
- **numpy**, **PyYAML**
- Optional: **trimesh** (mesh decimation path in `mesh_processing.py`)
- Feasibility pipeline additionally requires project IK stack (**EAIK**, etc.) and optionally **toppra** for TOPP phases

---

## Quick reference

| Goal | Entry point |
|------|-------------|
| Feasibility with collision (default) | `feasibility_analysis.py` + `batch_feasibility_config.yaml` |
| Disable collision | `--no-collision` |
| Offline C-space tests | `--cspace-forbidden-yaml` + `--cspace-only` |
| RS CSV geometric validation | `python tests/test_collision.py --internal-collision` |
| Stand-alone scene check | `SceneCollisionChecker.from_urdf_and_scene_yaml(...)` |
| Stand-alone self check | `SelfCollisionChecker(urdf_path=...).calibrate()` |
