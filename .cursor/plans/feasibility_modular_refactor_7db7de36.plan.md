---
name: Feasibility Modular Refactor
overview: Eliminate duplicated code across 3 singularity scripts, clean the feasibility pipeline into clearly separated phases, consolidate scattered config into grouped sections with a typed config dataclass, simplify config loading, and strip test_reachability down to pure reachability checking.
todos:
  - id: consolidate-singularity
    content: Merge core/singularity_analysis.py + core/unified_singularity.py + core/checks/singularity.py into one unified module at core/checks/singularity.py with SingularityAnalyzer supporting both 'unified' and 'classified' modes
    status: completed
  - id: strip-test-reachability
    content: Remove all singularity analysis from tests/test_reachability.py -- make it pure IK reachability pass/fail only (~400 lines)
    status: completed
  - id: feasibility-config-dataclass
    content: Create FeasibilityConfig dataclass to replace 38 function parameters in process_toolpath()
    status: completed
  - id: restructure-yaml
    content: "Restructure batch_feasibility_config.yaml: each group (singularity, manipulability, continuity, topp_ra, eaik, reachability) owns its own enabled + generate_graphs toggle; NO separate top-level graphs section"
    status: completed
  - id: simplify-config-loader
    content: Replace load_toolpath_config() passthrough hack + useless load_feasibility_config() with clean load_batch_config() returning FeasibilityConfig
    status: completed
  - id: clean-feasibility-pipeline
    content: "Refactor feasibility_analysis.py process_toolpath() into clearly labeled pipeline steps: load -> transform -> IK -> TOPPRA -> velocity -> analysis -> plots"
    status: completed
  - id: update-batch-script
    content: Simplify feasibility_analysis_batch.py to use FeasibilityConfig, remove fallback to nonexistent feasibility_config.yaml
    status: completed
  - id: update-test-singularity
    content: Update tests/test_singularity_only.py to import from consolidated core.checks.singularity
    status: completed
  - id: update-all-imports
    content: Update core/__init__.py, core/checks/__init__.py, core/feasibility_checks.py, utils/feasibility_plot.py, and all other import sites
    status: completed
  - id: clean-time-parameterization
    content: Make TOPP-RA mandatory (no enabled toggle), remove redundant compute_timestamps() and compute_timestamps_unified_pose(), rename utils/time_parameterization.py to waypoint density utility
    status: completed
  - id: delete-old-singularity
    content: Delete core/singularity_analysis.py and core/unified_singularity.py after migration is complete
    status: completed
isProject: false
---

# Feasibility & Test Code Modular Refactoring Plan

## Current Problems Identified

### 1. Three Singularity Implementations (must become one)

- `[core/singularity_analysis.py](core/singularity_analysis.py)` -- `SingularityAnalyzer` with type classification (shoulder/elbow/wrist), computes SVD, sigma_min, condition number, manipulability **independently**
- `[core/unified_singularity.py](core/unified_singularity.py)` -- `UnifiedSingularity` with sigma_min-only detection, duplicates SVD/sigma_min/cond/manipulability
- `[core/checks/singularity.py](core/checks/singularity.py)` -- standalone `compute_singularity_proximity`, `compute_condition_number`, etc. -- **also** doing SVD independently

All three do `np.linalg.svd(J)` and compute the same base metrics. The low-level helpers in `checks/singularity.py` overlap with both analyzer classes.

### 2. test_reachability Does Too Much

`[tests/test_reachability.py](tests/test_reachability.py)` (906 lines) runs IK **and** singularity analysis -- duplicating what feasibility already does. Per the user's intent: reachability test should **only** check "is this point reachable via IK? yes/no" and nothing else.

### 3. Config is Scattered and Bloated

`[config/batch_feasibility_config.yaml](config/batch_feasibility_config.yaml)` has settings scattered across 14 top-level keys. `[utils/config_loader.py](utils/config_loader.py)` has `load_toolpath_config()` doing ugly passthrough of 11+ keys, plus `load_feasibility_config()` that extracts just 2 keys.

### 4. `process_toolpath()` has 38 Parameters

`[feasibility_analysis.py](feasibility_analysis.py)` `process_toolpath()` takes 38 separate arguments instead of a config object.

### 5. test_singularity_only.py is a Niche Script

`[tests/test_singularity_only.py](tests/test_singularity_only.py)` duplicates singularity analysis using the same two analyzers. Should just import from the consolidated singularity module.

---

## Refactoring Plan

### Phase 1: Consolidate Singularity into One Module

**Goal:** Single source of truth for all singularity analysis at `core/checks/singularity.py`.

Merge `core/singularity_analysis.py` + `core/unified_singularity.py` + `core/checks/singularity.py` into one file:

```python
# core/checks/singularity.py -- THE singularity module
class SingularityMode(Enum):
    UNIFIED = "unified"
    CLASSIFIED = "classified"

@dataclass
class SingularityReport:
    """Universal report -- works for both modes."""
    is_singular: bool
    is_reachable: bool = True
    mode: SingularityMode = SingularityMode.UNIFIED
    # Common metrics (always computed)
    sigma_min: float = 0.0
    sigma_max: float = 0.0
    condition_number: float = np.inf
    manipulability: float = 0.0
    singular_values: np.ndarray = ...
    # Classified-mode extras (None when mode=UNIFIED)
    singularity_type: Optional[SingularityType] = None
    active_types: List[str] = ...
    wrist_metrics: Optional[Dict] = None
    shoulder_metrics: Optional[Dict] = None
    elbow_metrics: Optional[Dict] = None

class SingularityAnalyzer:
    """Single class with mode='unified' or mode='classified'."""
    def __init__(self, mode="unified", threshold=0.01, ...): ...
    def analyze(self, jacobian, joint_positions=None, fk_solver=None) -> SingularityReport: ...
    def analyze_trajectory(self, ...) -> List[SingularityReport]: ...
    @staticmethod
    def export_csv(reports, output_path): ...

# Keep low-level helpers for direct use by FeasibilityAnalyzer
def compute_singularity_proximity(jacobian) -> float: ...
def compute_condition_number(jacobian) -> float: ...
def compute_max_singular_value(jacobian) -> float: ...
def analyze_singularity_spectrum(jacobian) -> Dict: ...
```

- Delete `core/singularity_analysis.py` and `core/unified_singularity.py`
- Update `core/__init__.py` to export from the consolidated module
- All consumers (`test_reachability.py`, `test_singularity_only.py`, `feasibility_analysis.py`, `feasibility_checks.py`, plotting) import from `core.checks.singularity`

### Phase 2: Strip test_reachability to Pure Reachability

**Goal:** `test_reachability.py` answers ONE question: "Can the robot reach every waypoint?"

- Remove all singularity analysis code from `check_trajectory_reachability()` -- drop the `singularity_analyzer` and `fk_solver` parameters
- Remove `singularity_reports` from `TrajectoryResult`
- Remove singularity config section from `reachability_config.yaml`
- Remove singularity plotting calls and CSV export
- Remove `--export-singularity-graphs` CLI arg
- The script becomes ~400 lines (down from 906): load toolpath -> IK per waypoint -> report pass/fail
- If someone wants singularity + reachability, they use the feasibility pipeline

### Phase 3: TOPP-RA is Mandatory + Single Time Parameterization

**Goal:** TOPP-RA always runs. There is exactly ONE time-parameterized trajectory generated per trajectory, and everything downstream uses it.

**Current mess** -- three separate timing mechanisms:

- `core/topp_check.py` -- TOPP-RA (the real, mandatory one)
- `utils/time_parameterization.py` -- `compute_timestamps()` (arc-length / speed, redundant)
- `utils/math.py` -- `compute_timestamps_unified_pose()` (yet another timestamp computation)

**Fix:**

- TOPP-RA (`core/topp_check.py`) is the **only** time parameterization. It always runs after IK succeeds (no `enabled` toggle). Its `ToppraResult` is the single source of `t_samples`, `q_t`, `qdot_t`, `qddot_t`.
- `utils/time_parameterization.py` keeps only `compute_arc_lengths()`, `check_waypoint_density()`, and `interpolate_sparse_segments()` -- these are **pre-IK waypoint density checks**, not time parameterization. Rename the module or move these into a `waypoint_density` utility to avoid confusion.
- Remove `compute_timestamps()` from `utils/time_parameterization.py` and `compute_timestamps_unified_pose()` from `utils/math.py` -- they are dead/redundant now that TOPP-RA provides the real timing.
- In the YAML config, `topp_ra:` only has `generate_graphs:` (no `enabled:` since it always runs).

**Pipeline flow:**

```
IK -> joint_positions -> TOPP-RA -> ToppraResult (ONE object)
                                        |
                        +-------+-------+-------+-------+
                        |       |       |       |       |
                      C1 check  task-   joint   plots   report
                               space   traj
                               vel
```

### Phase 4: Clean Feasibility Pipeline (Clear Phases)

**Goal:** `feasibility_analysis.py` reads like a clear sequence of labeled steps.

Replace the 38-parameter `process_toolpath()` with a config dataclass:

```python
@dataclass
class CheckGroupConfig:
    """Base for any check group -- always has enabled + generate_graphs."""
    enabled: bool = True
    generate_graphs: bool = True

@dataclass
class SingularityConfig(CheckGroupConfig):
    mode: str = "unified"          # unified | classified
    threshold: float = 0.01
    type_thresholds: Dict = ...    # wrist/shoulder/elbow (classified only)
    check_j5_only: bool = True
    j5_threshold_deg: float = 0.76

@dataclass
class ManipulabilityConfig(CheckGroupConfig):
    warning: float = 0.001
    translational_warning: float = 0.001
    rotational_warning: float = 0.001
    directional_warning: float = 0.01

@dataclass
class ContinuityConfig(CheckGroupConfig):
    pose_scale_m_per_rad: float = 0.1
    safety_factor: float = 1.05
    default_speed_mm_s: float = 100.0

@dataclass
class TimeParamConfig(CheckGroupConfig):
    check_frequency_hz: float = 50.0
    max_gap_mm: float = 5.0
    interpolate_sparse: bool = False
    default_speed_mm_s: float = 100.0

@dataclass
class ToppRaConfig:
    generate_graphs: bool = True   # no 'enabled' -- TOPP-RA always runs

@dataclass
class ReachabilityConfig:
    generate_graphs: bool = True   # always runs, only toggle is graphs

@dataclass
class EaikMultiSolutionConfig(CheckGroupConfig):
    weights: Dict = ...            # c0, singularity, manipulability

@dataclass
class FeasibilityConfig:
    """All settings for a single feasibility run, loaded from YAML."""
    # Robot
    robot_name: str
    urdf_path: str
    reach_m: float
    velocity_limits_rad_s: np.ndarray
    accel_limits_rad_s2: np.ndarray
    
    # Toolpath
    toolpath_path: str
    use_base_frame: bool = False
    knife_translation_m: Optional[np.ndarray] = None
    knife_quaternion: Optional[np.ndarray] = None
    
    # Solver
    solver_type: str = "pin"
    
    # Check groups (each with enabled + generate_graphs + settings)
    reachability: ReachabilityConfig = ...
    singularity: SingularityConfig = ...
    manipulability: ManipulabilityConfig = ...
    continuity: ContinuityConfig = ...
    time_parameterization: TimeParamConfig = ...
    topp_ra: ToppRaConfig = ...
    eaik_multi_solution: EaikMultiSolutionConfig = ...
    
    # Output
    output_dir: str = "output/feasibility"
    level1_only: bool = True
    save_analysis: bool = True
    
    # Performance
    max_ik_failures_per_trajectory: int = 1
```

The pipeline uses each group's flags naturally:

```python
if config.singularity.enabled:
    _run_singularity_analysis(...)
    if config.singularity.generate_graphs:
        _plot_singularity(...)

if config.continuity.enabled:
    _run_continuity_checks(...)
    if config.continuity.generate_graphs:
        _plot_continuity(...)
```

Restructure the main pipeline function to be clearly phased:

```python
def process_toolpath(config: FeasibilityConfig) -> dict:
    # Step 1: Load and prepare
    solvers = _create_solvers(config)
    trajectories = _load_and_transform(config)
    
    # Step 2: Per-trajectory pipeline
    for traj in trajectories:
        # Phase 1: IK + C0 continuity
        ik_result = _run_ik_phase(traj, solvers, config)
        
        # Phase 2: TOPP-RA (ALWAYS runs -- single time parameterization)
        topp_result = _run_toppra(ik_result)  # produces ToppraResult used everywhere downstream
        
        # Phase 3: Checks (all consume the single ToppraResult)
        task_vel  = _check_task_space_velocity(topp_result)
        c1_result = _check_c1_continuity(topp_result, config.continuity)
        sing_result = _run_singularity(ik_result, config.singularity)
        manip_result = _run_manipulability(ik_result, config.manipulability)
        
        # Phase 4: Per-group graph generation (each group decides independently)
        _generate_graphs(ik_result, topp_result, sing_result, manip_result, c1_result, config)
        
        # Phase 5: Report
        _generate_report(results, config)
```

### Phase 4: Restructure Config YAML

**Goal:** `batch_feasibility_config.yaml` has logically grouped sections, each with a clear `enabled` toggle.

New structure:

```yaml
# ---- I/O ----
robots_to_use: ["IRB 1300-7/1.4"]
knife_poses_to_use: ["pose_1"]
toolpaths_folder: "Assets/Robot APCC/Toolpaths/Sample"
output_folder: "output/feasibility_batch"
use_base_frame: false

# ---- Solver ----
solver: "pin"

# ---- Performance ----
max_ik_failures_per_trajectory: 1

# ---- Output ----
output:
  level1_only: true
  save_analysis: true

# ---- Reachability ----
# Always runs (IK is the foundation). Graphs here = reachability plots.
reachability:
  generate_graphs: true       # reachability_per_waypoint, reachability_rate (aggregated)

# ---- EAIK Multi-Solution ----
eaik_multi_solution:
  enabled: true
  weights: { c0: 10.0, singularity: 1.0, manipulability: 0.5 }
  generate_graphs: true       # eaik_solutions_with_scores

# ---- Singularity ----
singularity:
  enabled: true
  mode: "unified"             # unified | classified
  threshold: 0.01
  type_thresholds: { wrist: 0.1, shoulder: 0.1, elbow: 0.1 }
  check_j5_only: true
  j5_threshold_deg: 0.76
  generate_graphs: true       # singularity_per_waypoint, singularity_per_trajectory (aggregated)

# ---- Manipulability ----
manipulability:
  enabled: true
  warning: 0.001
  translational_warning: 0.001
  rotational_warning: 0.001
  directional_warning: 0.01
  generate_graphs: true       # manipulability_per_waypoint, decomposed, directional,
                              # manipulability_per_trajectory (aggregated)

# ---- Continuity (C0 + C1) ----
continuity:
  enabled: true
  pose_scale_m_per_rad: 0.1
  safety_factor: 1.05
  default_speed_mm_s: 100.0
  generate_graphs: true       # c0_continuity_per_waypoint, c0_summary (aggregated),
                              # c1_continuity_dashboard, c1_summary (aggregated)

# ---- Time Parameterization & Waypoint Density ----
time_parameterization:
  enabled: true
  check_frequency_hz: 50.0
  max_gap_mm: 5.0
  interpolate_sparse: false
  default_speed_mm_s: 100.0
  generate_graphs: true       # waypoint_density_per_trajectory

# ---- TOPP-RA (always runs -- cannot be disabled) ----
topp_ra:
  generate_graphs: true       # topp_ra_velocity_profile, task_space_velocity,
                              # joint_space_trajectory, trajectory_3d_spline

# ---- Ranking (combinatorial search only) ----
ranking:
  safety_bin_size: 10.0
```

**Design principle:** There is NO separate top-level `graphs:` section. Each functional group owns its graphs. The rule is simple:

- If a group is `enabled: true` AND `generate_graphs: true`, all graphs belonging to that group are generated (per-trajectory whenever applicable, plus aggregated).
- If a group is `enabled: false`, the check itself is skipped (no computation, no graphs).
- If a group is `enabled: true` but `generate_graphs: false`, the check runs (numbers in the report) but no PNGs are saved.
- Per-trajectory graphs are always generated when `generate_graphs` is on (no separate per_trajectory/aggregated toggle needed -- if the data exists, both are produced).

Key changes vs. current config:

- **Killed the standalone `graphs:` section** -- graph toggles live inside each group
- **Killed `checks:` and `thresholds:`** top-level keys -- each group owns its `enabled` and thresholds
- Singularity config unified (was split across `checks:`, `thresholds:`, and undocumented CLI args)
- Manipulability config unified (was split across `checks:` and separate `manipulability:`)
- Every group follows the same pattern: `enabled` + settings + `generate_graphs`

### Phase 5: Simplify Config Loading

**Goal:** One clean loading function that returns a typed config object.

Replace `load_toolpath_config()` + `load_feasibility_config()` with:

```python
def load_batch_config(config_path: str) -> FeasibilityConfig:
    """Load batch_feasibility_config.yaml into a typed FeasibilityConfig."""
    raw = load_yaml(config_path)
    # Direct mapping from YAML groups to dataclass fields
    # No passthrough hacks, no fallback files
    return FeasibilityConfig(...)
```

- Delete `load_feasibility_config()` (only extracted 2 keys, useless)
- Simplify `load_toolpath_config()` to remove the 11-key passthrough loop
- `feasibility_analysis_batch.py` stops trying to load nonexistent `feasibility_config.yaml`

### Phase 6: Clean test_singularity_only.py

- Update imports to use the consolidated `core.checks.singularity.SingularityAnalyzer`
- Remove duplicate `SingularityReport`/`UnifiedSingularityReport` imports from deleted files
- The script remains as a standalone tool for joint-angle-only singularity testing (no IK), but now uses the single source of truth

### Phase 7: Update core/**init**.py and Imports

- Remove exports of deleted `SingularityAnalyzer` (old), `SingularityReport` (old), `UnifiedSingularity`, `UnifiedSingularityReport` from `core/__init__.py`
- Export the new unified `SingularityAnalyzer`, `SingularityReport`, `SingularityType`, `SingularityMode` from `core.checks.singularity`
- Update `core/checks/__init__.py` to export the new classes
- Update all import sites across the codebase

---

## Files Changed Summary


| Action       | File                                     | What                                            |
| ------------ | ---------------------------------------- | ----------------------------------------------- |
| **REWRITE**  | `core/checks/singularity.py`             | Consolidated singularity module (merge 3 files) |
| **DELETE**   | `core/singularity_analysis.py`           | Merged into checks/singularity.py               |
| **DELETE**   | `core/unified_singularity.py`            | Merged into checks/singularity.py               |
| **REWRITE**  | `config/batch_feasibility_config.yaml`   | Grouped config structure                        |
| **REWRITE**  | `utils/config_loader.py`                 | New `load_batch_config()`, remove bloat         |
| **REWRITE**  | `feasibility_analysis.py`                | FeasibilityConfig dataclass, clear phases       |
| **SIMPLIFY** | `feasibility_analysis_batch.py`          | Use FeasibilityConfig, remove fallback          |
| **SIMPLIFY** | `tests/test_reachability.py`             | Strip singularity, pure IK pass/fail            |
| **UPDATE**   | `tests/test_singularity_only.py`         | Import from consolidated module                 |
| **UPDATE**   | `core/__init__.py`                       | Update exports                                  |
| **UPDATE**   | `core/checks/__init__.py`                | Update exports                                  |
| **UPDATE**   | `core/feasibility_checks.py`             | Import from consolidated module                 |
| **UPDATE**   | `utils/feasibility_plot.py`              | Import from consolidated module                 |
| **UPDATE**   | `tests/configs/reachability_config.yaml` | Remove singularity section                      |


