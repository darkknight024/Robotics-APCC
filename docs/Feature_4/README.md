# Collision checking

Discrete configuration tests at a joint vector **q** (radians): robot self-contact and static cell obstacles. Geometry is Pinocchio with coal / hpp-fcl (broad-phase plus GJK/EPA on triangle meshes). The same boolean gate is used in two places:

1. **Feasibility / EAIK** — colliding CFX slots are excluded from branch selection so the emitted waypoint uses a collision-free IK solution when one exists.
2. **Experiment 25 validation** — `tests/test_collision.py` scores every CFX slot against RobotStudio labels and writes annotated CSVs.

Collision is checked at **waypoints**, not along the continuous path between them, unless you wrap a checker in `TrajectoryCollisionChecker` on a dense **q** sequence.

The earlier self-collision-only note is archived at `docs/old/collision_checker_developer_report.md`. This document is the current contract.

---

## 1. Design

EAIK can return up to eight configuration-family (CFX) solutions for one Cartesian pose. Several of those analytical folds are not robot-reachable (outside URDF joint limits) or fold the arm into itself. Treating “any finite **q**” as a valid collision query produced false `1`s on otherwise clear waypoints.

The gate therefore has two layers:


| Layer         | Question                                                      | If no                                               |
| ------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| **Active IK** | Finite 6-vector, not least-squares, inside URDF joint limits? | Slot is missing (`-1`). Collision is not evaluated. |
| **Geometry**  | Does `has_collision(q)` report contact?                       | Slot is clear (`0`) or colliding (`1`).             |


Only **active, collision-free** slots (`0`) enter mixed-branch / coverage-then-cost selection. Colliding active slots are skipped. Missing or out-of-limit slots are skipped. If every active slot collides, kinematics stay reachable but no joint vector is emitted (`solve_method = collision_all_branches`).

The URDF is the robot **without** a baked-in fixture mesh (`IRB_1300_1400_URDF.urdf`). TCP for IK/FK is `ee_link` from `config/fixture_config.yaml`. A fixture collision mesh is attached only when that entry has a non-empty `stl` (currently empty).

---

## 2. Algorithm

### 2.1 Build (once per checker)

```
URDF collision STLs
        │
        ▼
 Pinocchio model + GeometryModel
        │
        ├─ optional fixture STL on parent link (if fixture_config.stl is set)
        ├─ register robot–robot pairs
        ├─ drop successive links (kinematic tree distance ≤ 1)
        ├─ calibrate: drop remaining robot–robot pairs with hit rate ≥ 0.95
        └─ add robot–environment pairs from collision_objects.yaml
                (whitelist_pairs removed; env pairs are never auto-excluded)
```

**Successive-link filter.** Pairs whose parent joints are the same (tree distance 0) or parent/child (distance 1) are removed. That covers Link_i vs Link_{i+1} and flange vs fixture on `Link_6`. This is a kinematic-tree hop count, not `|j1 − j2|` on joint indices.

**Calibration.** Sample neutral, joint-range midpoint, and `n_samples` uniform random poses in URDF limits (default 32). A remaining **robot–robot** pair is dropped if it collides in ≥ 95% of those samples. That removes visual-STL overlap at joints without requiring a hit on every random fold. Environment pairs are never dropped this way.

**Scene poses.** Environment STLs are fixed in the robot base frame. YAML uses `position_mm` and `quat_wxyz` (`[qw, qx, qy, qz]`). `collision: false` skips an object entirely.

**Margins.** Per-geometry `collision_tolerance_m` is applied as FCL `security_margin` on each pair (inflation before contact).

### 2.2 Query

For a configuration **q**:

1. Pad **q** to `model.nq`.
2. Forward-place all collision bodies.
3. Test only registered pairs.
4. `has_collision(q)` — `pin.computeCollisions(..., stop_at_first_collision=True)` (boolean, fast).
5. `check(q)` — also `computeDistances` for colliding pairs, closest pair, and minimum distance.

C-space checking (offline tests only) ignores meshes: **q** is in collision if it lies in a configured axis-aligned box in joint space.

### 2.3 Per-CFX report encoding

Shared by feasibility and Experiment 25 (`core/feasibility/collision_gate.py`):


| Value  | Meaning                                                                     |
| ------ | --------------------------------------------------------------------------- |
| **1**  | Active IK **and** `has_collision(q)` is true                                |
| **0**  | Active IK **and** collision-free                                            |
| **−1** | Missing slot, non-finite **q**, least-squares, or outside URDF joint limits |


Collision is never evaluated for `−1`. Out-of-limit analytical folds (typical extra `1`s on CFX 4/5 or 6/7) must not be reported as collisions.

---

## 3. EAIK branch gating

EAIK enumerates CFX slots. The collision checker is **not** attached to the IK solver. Feasibility owns the gate.

```
Pose (xyz m, quaternion)
        │
        ▼
 EAIK: up to 8 CFX joint vectors
        │
        ▼
 For each slot:
   not finite / LS / out of limits  →  skip (−1)
   has_collision(q)                 →  skip (blocked)
   else                             →  eligible for selection
        │
        ▼
 Mixed / coverage-then-cost CFX selection
   scores only eligible (collision-free, in-limit) slots
        │
        ├── at least one eligible slot → emit that q
        └── all active slots collide   → keep is_reachable,
                                         joint_positions_rad = None,
                                         reason = collision_all_branches
```

Level-1 feasibility pass then requires `collision_ok` when a checker is attached (no selected-path collision, no all-branches collision, no output leak).

Disable for a run with `--no-collision`. Production default is **on** (`config/batch_feasibility_config.yaml` → `collision.enabled: true`).

---

## 4. Configuration


| File                                             | Role                                                                                                            |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `config/collision_config.yaml`                   | Exp25 / checker defaults: URDF, `fixture_name`, `ee_frame_name`, `scene_yaml`, toolpath and results directories |
| `config/fixture_config.yaml`                     | TCP origin vs `Link_6`; optional fixture `stl`                                                                  |
| `config/collision_objects.yaml`                  | Cell meshes (pedestal, scanner, Zund knife, …)                                                                  |
| `config/collision_objects_empty.yaml`            | Self pairs only (no env meshes)                                                                                 |
| `config/robots_config.yaml`                      | URDF path, `fixture_name`, joint limits                                                                         |
| `config/batch_feasibility_config.yaml`           | `collision:` block for feasibility / EAIK                                                                       |
| `config/cspace_forbidden_zones_irb1300_714.yaml` | Joint-space boxes for offline tests                                                                             |


Fixture vs URDF: keep `IRB_1300_1400_URDF.urdf` (no fixture mesh in the URDF). Change TCP or fixture STL in `fixture_config.yaml` / `collision_config.yaml`, not via a CLI flag.

---

## 5. Known limitations

- **Mesh fidelity.** URDF and cell STLs could be coarser or looser than ABB RobotStudio’s internal model. Agreement with RS is empirical.
- **Discrete waypoints only.** A collision-free start and end does not prove the motion between them is clear, unless you densely sample **q** and use `TrajectoryCollisionChecker`.
- **Calibration is a heuristic.** A pair that always collides by design would be excluded. A pair that collides in 90% of samples is kept and can still false-positive. Environment pairs are never auto-excluded, so a badly placed cell mesh will flag many waypoints.

---

## 6. Running `tests/test_collision.py`

Default mode is **Experiment 25**: IK every pose in `toolpaths_dir`, score eight CFX flags, compare the filename CFX (`*_cfxN.csv`) to the last-column RobotStudio label.

Setup is `config/collision_config.yaml` (URDF, fixture, scene, paths).

```bash
# Experiment 25 (default) — writes Results/<MM_DD_YY_HH_MM_SS>/
python tests/test_collision.py

# Same, explicit config
python tests/test_collision.py --collision-config config/collision_config.yaml

# Optional overrides
python tests/test_collision.py --exp25-dir Robot_APCC/Experiments/Experiment_25/Toolpaths
python tests/test_collision.py --scene-yaml config/collision_objects.yaml

# Import smoke only (no dataset)
python tests/test_collision.py --smoke

# Internal_Collision RS joint CSVs (legacy geometric split)
python tests/test_collision.py --internal-collision
```

Requires Pinocchio/coal, numpy, PyYAML, and the EAIK stack. Run from the repo root so URDF and STL paths resolve.

**Experiment 25 input contract**

- Headerless T0 files: preamble (`1`, `T0`, count), then pose rows.
- Columns 1–7: xyz **millimetres** + quaternion in robot base (`T_B_K`). xyz is divided by 1000 for EAIK/Pinocchio.
- Columns 8–14: ignored (speed/zone).
- Last original column: GT `0` / `1` for the CFX in the filename (all current files are `*_cfx0.csv`).
- Each file typically has one GT=`1` row (usually the last).

---

## 7. Interpreting Experiment 25 results

Each run writes:

```
Robot_APCC/Experiments/Experiment_25/Results/<MM_DD_YY_HH_MM_SS>/
  summary.txt
  exp25_metrics.json
  summary.md / summary.json
  collision_with_scene_objects/*.csv
  self_collisions/*.csv
```

**Annotated CSVs** keep the exact input row and append `cfx0..cfx7` (`1` / `0` / `−1` as in §2.3).

Example (Zund knife, clear waypoint vs last colliding waypoint):

```
…,0,  0,0,-1,-1,-1,-1,-1,-1     # GT=0; CFX 0/1 active and clear
…,1,  1,1,-1,-1,-1,-1,-1,-1     # GT=1; CFX 0/1 collide with knife / POT / plates
```

`**summary.txt**`

- **filename CFX vs GT** — predicted flag on the CFX named in the file vs the GT column. This is the pass/fail metric (`fp = 0` and `fn = 0`).
- **any CFX branch vs GT** — `1` if **any** active slot collides. This is the Feature 2 gate and will over-predict vs a single-CFX RS label when unused families fold into the arm.

A slot that is `−1` is omitted from that comparison (not counted as a false negative). Pedestal GT=`1` may live on CFX 6/7 with CFX 0 unusable; filename-CFX `tp` can then be 3 of 4 files while any-branch still catches the hit.

**Expected pattern after joint-limit gating**

- Clear rows: `0` on existing in-limit families, `−1` elsewhere — not `1` on out-of-limit folds.
- The labelled collision row: `1` on the active in-limit family that actually contacts the named object (or self).

---

## 8. Feasibility pipeline

```bash
python feasibility_analysis.py -t path/to/toolpath.csv -u path/to/robot.urdf
# collision on unless:
python feasibility_analysis.py ... --no-collision
```

Per trajectory: `collision_check_enabled`, `collision_ok`, `collision_reject_count`, `collision_all_branches_count`, `collision_output_leak_count` (should be 0). `analysis_report.txt` includes Collision PASS/FAIL when checking is on.

**C-space-only** (no meshes, offline):

```bash
python feasibility_analysis.py -t ... -u ... \
  --cspace-forbidden-yaml config/cspace_forbidden_zones_irb1300_714.yaml \
  --cspace-only
```

**Library**

```python
from core.collision import SceneCollisionChecker, build_collision_checker_for_feasibility

checker = SceneCollisionChecker.from_urdf_and_scene_yaml(
    urdf_path="Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF.urdf",
    scene_yaml_path="config/collision_objects.yaml",
    calibrate=True,
    fixture_name="ee_link",
)
if checker.has_collision(q):
    print(checker.check(q).colliding_pairs)
```

`has_collision(q) -> bool` with `True` = infeasible is the contract for any gate passed into `FeasibilityAnalyzer`.

---

## 9. Module map


| Location                                   | Role                                                                      |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| `geometry.py`                              | URDF collision model; fixture STL; env poses (`position_mm`, `quat_wxyz`) |
| `pair_rules.py`                            | Self/env pairs; tree-distance adjacency; hit-rate calibration             |
| `scene_checker.py`                         | Self + environment                                                        |
| `self_checker.py`                          | URDF self only                                                            |
| `factory.py`                               | `build_collision_checker_for_feasibility`                                 |
| `core/feasibility/collision_gate.py`       | Active IK + `{-1, 0, 1}` flags                                            |
| `core/feasibility/cfx_branch_selection.py` | Scores only collision-free in-limit slots                                 |
| `core/feasibility/analyzer.py`             | Annotate CFX; drop selected **q** if all branches collide                 |
| `utils/feasibility/pipeline_runner.py`     | Builds checker from batch config + CLI                                    |


**Legacy shim:** `core/collision_checker.py` re-exports `SelfCollisionChecker`.

---

## Quick reference


| Goal                         | Command / entry                                       |
| ---------------------------- | ----------------------------------------------------- |
| Exp25 validation             | `python tests/test_collision.py`                      |
| Setup (URDF, fixture, scene) | `config/collision_config.yaml`                        |
| Feasibility with collision   | `feasibility_analysis.py` (default on)                |
| Disable collision            | `--no-collision`                                      |
| Stand-alone scene check      | `SceneCollisionChecker.from_urdf_and_scene_yaml(...)` |

## Next Steps
- Test with end effector fixture stl
- Test in valid paths, but some of the cfx are in collision.
- Calibrate collision exclusion with more data (optional)
- Test with midsole mounted on the fixture and being very close to knife
- Time profile and decimate meshes wherever needed