# Feature 4 — Collision (`core.collision`)

Pinocchio + **coal** collision geometry for:

- **Self-collision** — `SelfCollisionChecker` (URDF collision meshes only; same behaviour as the legacy `core/collision_checker.py` shim).
- **Full scene** — `SceneCollisionChecker.from_urdf_and_scene_yaml(urdf, config/collision_objects.yaml)` loads static STLs fixed in the robot base frame. Object **poses** use **`xyz` in mm** and **`quaternion` [qw,qx,qy,qz]** (see `geometry.se3_from_collision_object_pose`). Optional **`mesh_scale`** (e.g. mm STLs), convex parts, per-object `collision_tolerance_m` (via `security_margin`), and **whitelist** pairs.

Supporting modules:

| Module | Role |
|--------|------|
| `scene_config.py` | Parse `collision_objects.yaml` |
| `geometry.py` | URDF geom load, `SE3` from pose dict, append fixed meshes |
| `mesh_processing.py` | Precomputed simplified STL preferred; optional **trimesh** decimation |
| `pair_rules.py` | Self pairs, adjacency filter, robot–environment pairs |
| `object_checker.py` | `ObjectCollisionChecker` — robot vs environment subset |
| `midsole_checker.py` | `MidsoleCollisionChecker` — optional midsole policy diagnostics |
| `trajectory_checker.py` | Dense `q` path sweep |

Feasibility integration: pass a checker with `has_collision(q) -> bool` into `FeasibilityAnalyzer(..., collision_checker=...)`. Colliding EAIK `cfx` slots are rejected before scoring.
