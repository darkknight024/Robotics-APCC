# Robotics-APCC — Live Interactive Visualizer

> **README for Coding Agent**
> This document describes everything needed to build the Live Interactive Visualizer for the Robotics-APCC project. Read every section before writing a single line of code. Where it says "read file X first", do that before continuing.

---

## Table of Contents

1. [What You Are Building](#1-what-you-are-building)
2. [Before You Start — Read These First](#2-before-you-start--read-these-first)
3. [Project File Map (Existing Codebase)](#3-project-file-map-existing-codebase)
4. [Frontend Tech Stack](#4-frontend-tech-stack)
5. [Application Architecture — One App, Two Layers](#5-application-architecture--one-app-two-layers)
6. [Directory Structure of the New Visualizer](#6-directory-structure-of-the-new-visualizer)
7. [User Workflow — CSV Upload Pipeline](#7-user-workflow--csv-upload-pipeline)
8. [Data Detection Logic](#8-data-detection-logic)
9. [Frame & Robot Configuration Step](#9-frame--robot-configuration-step)
10. [Analysis Mode Selection](#10-analysis-mode-selection)
11. [Analysis Modes — What Each One Does](#11-analysis-modes--what-each-one-does)
12. [Feasibility Analysis Mode](#12-feasibility-analysis-mode)
13. [Plot System — Groups, Timeline Cursors, ECFX Branches](#13-plot-system--groups-timeline-cursors-ecfx-branches)
14. [ECFX Branch Classification (EAIK Solutions)](#14-ecfx-branch-classification-eaik-solutions)
15. [3D Visualization — Viser Embedding](#15-3d-visualization--viser-embedding)
16. [Config as UI Items](#16-config-as-ui-items)
17. [TeleOp Tab](#17-teleop-tab)
18. [Backend Python Server](#18-backend-python-server)
19. [API Contract — Frontend ↔ Backend](#19-api-contract--frontend--backend)
20. [Implementation Phases](#20-implementation-phases)
21. [Important Constraints & Rules](#21-important-constraints--rules)

---

## 1. What You Are Building

A **single web application** called the **Robotics-APCC Live Visualizer**. It is built on top of the existing Robotics-APCC kinematic analysis project and adds an interactive, browser-based interface for everything the project does.

The application has **two primary areas**:

| Area | Description |
|------|-------------|
| **Analysis Tab** | Upload a CSV, detect what data is in it, run FK/IK/feasibility analysis using the existing Python pipeline, view results live in 3D with synchronized plot panels |
| **TeleOp Tab** | Drive the robot live in the browser using keyboard/mouse in task space or joint space, record trajectories, export them |

**Core principle:** The existing Python scripts in `Robotics-APCC/` are **never modified**. They are orchestrators. The visualizer calls them as subprocesses or imports their internal functions directly. All the logic already exists — this project is the interactive front-end wrapper around it.

---

## 2. Before You Start — Read These First

Before writing any code, the coding agent must read the following files from the existing codebase. These files define the data formats, available functions, and config structures that the visualizer must use.

### 2.1 Mandatory Pre-Reading

```
utils/config_loader.py          ← Understand FeasibilityConfig, RobotConfig, KnifePose.
                                  This defines ALL config parameters and their types.
                                  All UI config items come from here.

utils/csv_loader_toolpath.py    ← Understand all CSV column formats the system accepts.
                                  This is the source of truth for data detection logic.
                                  Understand T0 separator, header styles, rs_* aliases.

utils/csv_loader_robostudio.py  ← Understand RobotStudio CSV format.
                                  Columns: rs_x_mm, rs_y_mm, rs_z_mm, rs_qw..qz,
                                  rs_j1_deg..rs_j6_deg, is_reachable.

tests/test_solvers.py           ← This is the FK/IK comparison orchestrator.
                                  Read it to understand what comparisons are run,
                                  what output files it produces, and what plots it needs.
                                  Do NOT reimplement its logic. Call it or import from it.

feasibility_analysis.py         ← Read this for the feasibility pipeline.
                                  Understand process_toolpath() — its input parameters,
                                  what it computes (IK→TOPP-RA→C1→singularity→manipulability),
                                  and what files it outputs. Do NOT reimplement.

utils/generate_plot_ik.py       ← Read to understand what IK comparison plots are needed.
                                  You will NOT generate static PNG files. Instead, use this
                                  file to understand what data series each plot requires,
                                  then render those as live Plotly charts in the frontend.

utils/feasibility_plot.py       ← Same as above but for feasibility plots.
                                  Read to understand all plot types and their data requirements.
                                  Render these as live Plotly/uPlot charts, not PNGs.

core/eaik_ik_solver.py          ← Read to understand ECFX, multi-solution mode, how
                                  compute_ecfx() works, and what IK_Solution returns.

core/__init__.py                ← Read to understand create_solvers() factory.
```

### 2.2 Read for Context (Not Mandatory Before Starting)

```
core/feasibility_checks.py      ← FeasibilityAnalyzer, score_ik_solution_breakdown
core/checks/singularity.py      ← SingularityAnalyzer, SingularityReport
core/checks/manipulability.py   ← Yoshikawa and decomposed manipulability
core/topp_check.py              ← ToppraResult structure
config/robots_config.yaml       ← All robot names — used to populate robot selector
config/knife_config.yaml        ← All knife pose names — used to populate knife selector
```

---

## 3. Project File Map (Existing Codebase)

> Do not modify any of these files. Use them as a library.

### Root Scripts (Orchestrators — call these via subprocess or import)

| File | Role in Visualizer |
|------|-------------------|
| `feasibility_analysis.py` | Call `process_toolpath()` for feasibility analysis mode |
| `feasibility_analysis_batch.py` | Not directly used in visualizer (batch headless only) |
| `combinatorial_search.py` | Results can be loaded and browsed in a future mode |
| `ik_solver_benchmark.py` | Optional: call for solver timing comparison |
| `visualize_robot.py` | Reference implementation of Viser + URDF viewer — study this |

### Core (Import directly — do not subprocess these)

| Module | What to import |
|--------|---------------|
| `core.__init__` | `create_solvers(urdf_path, solver, ik_config)` |
| `core.base_solvers` | `BaseFKSolver`, `BaseIKSolver`, `FKResult` type annotations |
| `core.pin_fk_solver` | `PinocchioFKSolver` |
| `core.pin_ik_solver` | `PinocchioIKSolver` |
| `core.eaik_fk_solver` | `EAIKFKSolver` |
| `core.eaik_ik_solver` | `EAIKIKSolver`, `compute_ecfx`, `ECFXLabel` |
| `core.feasibility_checks` | `FeasibilityAnalyzer`, `score_ik_solution_breakdown` |
| `core.topp_check` | `parameterize_trajectory`, `ToppraResult` |
| `core.collision_checker` | `CollisionChecker` |
| `core.checks.singularity` | `SingularityAnalyzer`, `SingularityReport` |
| `core.checks.manipulability` | `compute_manipulability`, decomposed variants |
| `core.checks.c0_continuity` | C0 joint jump checks |
| `core.checks.c1_continuity` | `C1Result` |
| `core.checks.task_space_velocity` | Cartesian speed limit checks |

### Utils (Import directly)

| Module | What to import |
|--------|---------------|
| `utils.config_loader` | `load_robots_config`, `load_knife_config`, `load_batch_config`, `FeasibilityConfig` |
| `utils.urdf_loader` | `load_urdf` |
| `utils.csv_loader_toolpath` | `load_toolpath_trajectories_ext` |
| `utils.csv_loader_robostudio` | `load_robostudio_full`, `validate_robostudio_csv` |
| `utils.transform_handler` | `transform_trajectories_to_base_frame`, `compute_ecfx` |
| `utils.math` | `shortest_angle`, joint distance helpers |
| `utils.time_parameterization` | `check_waypoint_density`, `interpolate_sparse_segments` |

---

## 4. Frontend Tech Stack

### 4.1 Required Libraries

Research and use the following. These are chosen for minimal bundle size, unique visual design, and the specific needs of this project.

```
Framework:      React 18 + TypeScript
Build:          Vite 5
Styling:        Tailwind CSS v3
Components:     shadcn/ui  (Radix UI primitives — accessible, unstyled, composable)
Charts:         Plotly.js via react-plotly.js
                  ↳ Required for: synchronized cursor annotations across all plot panels,
                    rich line charts, heatmaps, linked axis zoom
                  ↳ DO NOT use Recharts or Chart.js — they lack Plotly's annotation/shape
                    system which is required for the timeline cursor mechanism
State:          Zustand  (minimal, no boilerplate)
Routing:        React Router v6  (for tab navigation: /analysis, /teleop)
File upload:    react-dropzone
CSV parsing:    Papa Parse  (browser-side column detection before sending to backend)
3D viewer:      Viser embedded via <iframe> (see Section 15)
Websocket:      native browser WebSocket API  (for streaming backend events)
HTTP client:    Axios or native fetch
Icons:          lucide-react
Notifications:  sonner  (minimal toast library)
Splitter:       react-resizable-panels  (for resizable left/center/right layout)
```

### 4.2 What NOT to use

- `@mui/material` or `Ant Design` — too heavy, generic look
- `Recharts` or `Chart.js` — missing shape annotation API needed for timeline cursors
- `Three.js` directly — handled by Viser
- Any CSS-in-JS runtime library (styled-components, emotion) — use Tailwind only

### 4.3 Design Philosophy

- **Dark theme by default** — robotics/engineering tools live in dark UI
- **Minimal chrome** — the 3D viewer and plots are the content; UI controls are secondary
- **Dense but readable** — lots of data visible at once; use small font sizes with good contrast
- **No splash screens, no animations on load** — engineering tool, not a marketing page
- **Monospace font for all numeric data** — `font-mono` class via Tailwind

---

## 5. Application Architecture — One App, Two Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser (React + Vite)                        │
│                                                                  │
│  ┌──────────┐  ┌────────────────────────────────────────────┐   │
│  │ Analysis │  │              TeleOp Tab                     │   │
│  │  Tab     │  │  keyboard → WebSocket → Python IK loop     │   │
│  │          │  │  3D robot (Viser iframe) updates live       │   │
│  │ CSV upload│ └────────────────────────────────────────────┘   │
│  │ detect   │                                                    │
│  │ configure│  ┌────────────────────────────────────────────┐   │
│  │ analyze  │  │     Viser iframe (embedded .viser stream)   │   │
│  │ plots    │  │     served from Python Viser server         │   │
│  └──────────┘  └────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP REST + WebSocket
┌───────────────────────────▼─────────────────────────────────────┐
│              Python Backend (FastAPI)                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Visualizer Server (visualizer/backend/server.py)         │   │
│  │                                                          │   │
│  │  • /api/upload          CSV upload + data detection      │   │
│  │  • /api/detect-columns  column sniffing                  │   │
│  │  • /api/transform       T_P_K → T_B_P transform          │   │
│  │  • /api/run-ik          run IK on task-space waypoints   │   │
│  │  • /api/run-fk          run FK on joint-space waypoints  │   │
│  │  • /api/compare         FK/IK compare (test_solvers)     │   │
│  │  • /api/feasibility     run process_toolpath()           │   │
│  │  • /api/results/{id}    poll/stream results              │   │
│  │  • WS /ws/stream        stream logs + partial results    │   │
│  │  • WS /ws/teleop        keyboard → IK → robot state      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Viser Server (visualizer/backend/viser_server.py)        │   │
│  │  Serves the 3D robot scene on port 8081                  │   │
│  │  Receives scene-update events from the main server       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Imports from Robotics-APCC (never modified):             │   │
│  │  core.*, utils.*, feasibility_analysis.process_toolpath  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key point:** The FastAPI server and the Viser server are two separate Python processes but they share state via an internal queue/pipe. FastAPI handles all REST/WS from the frontend. Viser handles only the 3D scene and is embedded as an iframe.

---

## 6. Directory Structure of the New Visualizer

```
Robotics-APCC/                        ← existing project root (do not touch)
  core/
  utils/
  config/
  Assets/
  feasibility_analysis.py
  ...

visualizer/                            ← NEW: entire visualizer lives here
  backend/
    server.py                          ← FastAPI main server (port 8080)
    viser_server.py                    ← Viser 3D scene server (port 8081)
    scene_state.py                     ← shared SceneState dataclass
    data_detection.py                  ← CSV column sniffing logic
    pipeline_runner.py                 ← async wrappers around existing scripts
    teleop/
      teleop_handler.py                ← keyboard → IK → robot update loop
      recorder.py                      ← waypoint capture + trajectory save
      live_metrics.py                  ← per-keypress metric computation
    config/
      teleop_config.yaml               ← TeleOp-specific config (step sizes, keybinds)
  frontend/
    package.json
    vite.config.ts
    tailwind.config.ts
    src/
      main.tsx
      App.tsx                          ← router: /analysis, /teleop
      stores/
        analysisStore.ts               ← Zustand store for analysis state
        teleopStore.ts                 ← Zustand store for teleop state
      components/
        layout/
          AppShell.tsx                 ← top nav, tab switching, global layout
          ResizableLayout.tsx          ← left panel | center | right panel
        analysis/
          UploadStep.tsx               ← drag-drop CSV upload
          DetectStep.tsx               ← column detection + unknown column mapper
          FrameStep.tsx                ← base frame vs knife frame, knife selector
          RobotStep.tsx                ← robot selector
          ActionStep.tsx               ← mode selection (IK/FK/Compare/Feasibility)
          ConfigPanel.tsx              ← dynamic config UI from FeasibilityConfig
          RunPanel.tsx                 ← run button + streaming log
          ResultsSummary.tsx           ← success/failure summary after run
        plots/
          PlotGroup.tsx                ← collapsible group containing multiple plots
          TimelinePlot.tsx             ← single Plotly plot with cursor annotation
          EcfxBranchPlot.tsx           ← ECFX colored branch plots (see Section 14)
          PlotDashboard.tsx            ← all plot groups rendered together
        viewer/
          ViserEmbed.tsx               ← <iframe> wrapping the Viser server
          TimelineBar.tsx              ← bottom timeline scrubber + playback controls
        teleop/
          TeleopPanel.tsx              ← left control panel (mode, step size, home)
          KeyboardMap.tsx              ← visual keyboard reference + clickable buttons
          HudMetrics.tsx               ← live manipulability/condition/joint gauges
          PoseInput.tsx                ← target pose text input + solve button
      hooks/
        useTimeline.ts                 ← timeline state + cursor broadcast
        useWebSocket.ts                ← WS connection management
        usePlotCursors.ts              ← batch-update all Plotly cursor annotations
      types/
        data.ts                        ← TypeScript types for all API responses
        ecfx.ts                        ← ECFXLabel type
```

---

## 7. User Workflow — CSV Upload Pipeline

This is the main user flow in the Analysis tab. Every step is a discrete UI state. The user moves forward through steps; they can go back to any previous step.

```
Step 1: Upload
   User drags or selects a CSV file (or a directory of CSVs)
   → Papa Parse reads the first 20 rows in-browser for column sniffing
   → POST /api/upload with file

Step 2: Detect Data
   Backend sniffs columns (read csv_loader_toolpath.py and csv_loader_robostudio.py
   to understand all valid column names and aliases)
   Backend returns DetectionResult:
     { has_task_space: bool, has_joint_space: bool,
       detected_columns: {...}, unknown_columns: [...] }
   
   If unknown_columns is non-empty:
     Show column mapper UI: user assigns each unknown column to a known role
     (x_mm, y_mm, z_mm, qw, qx, qy, qz, j1_deg..j6_deg)
     This mapping is sent back so the loader can handle it.

Step 3: Frame Configuration  (only if has_task_space = true)
   Ask: "Is your task-space data already in the robot base frame?"
   Options:
     [ ] Already in base frame (use_base_frame = true)
     [ ] In knife frame (T_P_K) — select knife pose:
         <dropdown populated from knife_config.yaml>
   
   If knife frame selected:
     Backend calls transform_trajectories_to_base_frame() immediately
     and returns transformed waypoints for preview in the 3D viewer.

Step 4: Robot Selection  (always shown)
   Dropdown populated from robots_config.yaml
   Shows robot name, URDF path, reach, and velocity limits
   
   Selecting a robot:
     → Backend loads the URDF into the Viser server
     → Viser iframe updates to show the selected robot at home pose

Step 5: Choose What To Do
   Show only the options valid for the detected data:
   
   If has_task_space AND has_joint_space:
     ✓ FK/IK Comparison  (compare our solver vs loaded joint data vs RobotStudio)
     ✓ Run IK only        (solve IK on task-space data)
     ✓ Run FK only        (run FK on joint-space data)
     ✓ Feasibility Analysis  (full pipeline)
   
   If has_task_space only:
     ✓ Run IK only
     ✓ Feasibility Analysis
   
   If has_joint_space only:
     ✓ Run FK only
   
   None of the above includes TeleOp — that lives on its own tab.

Step 6: Configure
   Show dynamic config UI relevant to the selected mode.
   Each config parameter comes directly from FeasibilityConfig / ik_config.yaml.
   See Section 16 for how config becomes UI.

Step 7: Run
   Run button triggers the appropriate backend call.
   A streaming log panel shows stdout in real time via WebSocket.
   After completion: results load into the plot groups and 3D viewer.
```

---

## 8. Data Detection Logic

> Read `utils/csv_loader_toolpath.py` and `utils/csv_loader_robostudio.py` fully before implementing this.

The backend `data_detection.py` module must correctly identify what data is present in a CSV. Below is the detection logic:

### 8.1 Task-Space Detection

A CSV has **task-space data** if it contains position AND orientation columns. Accepted column names (read the loader to get the exact list of aliases):

```python
# Position (any of these sets count)
position_cols = {
  'x', 'x_mm', 'rs_x_mm', 'pos_x', 'tcp_x',
  'y', 'y_mm', 'rs_y_mm', 'pos_y', 'tcp_y',
  'z', 'z_mm', 'rs_z_mm', 'pos_z', 'tcp_z'
}

# Orientation quaternion
orientation_cols = {
  'qw', 'rs_qw', 'qx', 'rs_qx', 'qy', 'rs_qy', 'qz', 'rs_qz'
}
```

If the CSV has any recognized position set AND any recognized quaternion set → `has_task_space = True`.

### 8.2 Joint-Space Detection

A CSV has **joint-space data** if it contains 6 joint angle columns:

```python
joint_cols = {
  ('j1_deg', 'j2_deg', 'j3_deg', 'j4_deg', 'j5_deg', 'j6_deg'),
  ('rs_j1_deg', 'rs_j2_deg', 'rs_j3_deg', 'rs_j4_deg', 'rs_j5_deg', 'rs_j6_deg'),
  ('joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'),
  # ... read the loader for all accepted forms
}
```

### 8.3 Unknown Column Handling

Any column not recognized by the above patterns goes into `unknown_columns`. The UI shows a mapper:

```
Column "theta_1"  →  [dropdown: j1_deg | j2_deg | ... | x_mm | ... | ignore]
Column "pos_A"    →  [dropdown: x_mm | y_mm | z_mm | ... | ignore]
```

The user's mapping is stored and passed back to the loader via a `column_map` parameter.

### 8.4 Multi-Trajectory Detection

The loader splits CSV into sub-trajectories at rows where the first column equals `T0` or where there is a trajectory separator header. The detection returns `num_trajectories` so the UI can show a trajectory selector immediately.

---

## 9. Frame & Robot Configuration Step

### 9.1 Knife Frame Transform

When the user selects a knife pose, the backend immediately calls:

```python
from utils.transform_handler import transform_trajectories_to_base_frame
from utils.config_loader import load_knife_config

knife_config = load_knife_config("config/knife_config.yaml")
knife = knife_config[selected_knife_name]

transformed = transform_trajectories_to_base_frame(
    trajectories_t_p_k,
    knife_translation_m=knife.translation_m,
    knife_quaternion=knife.quaternion
)
```

The transformed waypoints are stored in session state and used for all subsequent analysis.

### 9.2 Robot Loading

When a robot is selected, the backend:

1. Calls `utils.urdf_loader.load_urdf(robot_config)` to get the `RobotModel`
2. Calls `create_solvers(urdf_path, solver="pin", ik_config=...)` to instantiate FK/IK solvers
3. Sends a scene update to the Viser server via internal queue to load the URDF and show the robot at home pose
4. Returns robot metadata to the frontend (reach, joint limits, joint names)

---

## 10. Analysis Mode Selection

### Mode: FK/IK Comparison

Available when: `has_task_space AND has_joint_space`

What it does:
- Takes each waypoint's joint angles → runs FK → gets computed TCP
- Takes each waypoint's TCP pose → runs IK → gets computed joint angles
- Compares both against what was in the file (which acts as "ground truth")
- This is what `tests/test_solvers.py` does — read that script to understand the full comparison logic and what output data is produced
- Do NOT re-implement the comparison logic — call `test_solvers.py` via subprocess or import its comparison functions directly

Output data needed for plots (read `utils/generate_plot_ik.py` for the full list):
- FK position error (Euclidean distance per waypoint)
- FK per-axis deltas (Δx, Δy, Δz)
- IK joint angle deltas (Δj1..Δj6 per waypoint)
- IK success/failure per waypoint
- EAIK: ECFX label per solution, all solution branches (see Section 14)
- Pinocchio: solve method per waypoint (initial_guess / neutral / random / failed)

### Mode: IK Only

Available when: `has_task_space`

What it does: Runs IK solver on every task-space waypoint. Returns joint angles + success/failure + ECFX labels (if EAIK). The result is immediately shown in the 3D viewer and plots.

### Mode: FK Only

Available when: `has_joint_space`

What it does: Runs FK on every joint-space waypoint. Returns TCP positions and orientations. These are drawn as a 3D path in the viewer.

### Mode: Feasibility Analysis

Available when: `has_task_space` (IK must be solvable to run feasibility)

See Section 12 for full detail.

---

## 11. Analysis Modes — What Each One Does

### 11.1 Calling Existing Scripts

**Do not reimplement any analysis logic.** Use this pattern:

```python
# Option A: import the function directly (preferred for IK/FK single runs)
import sys
sys.path.insert(0, "/path/to/Robotics-APCC")
from feasibility_analysis import process_toolpath
from core import create_solvers

# Option B: subprocess (for long-running analysis with log streaming)
import asyncio

async def run_feasibility_streamed(config_args, ws_send):
    proc = await asyncio.create_subprocess_exec(
        "python", "feasibility_analysis.py", *config_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd="/path/to/Robotics-APCC"
    )
    async for line in proc.stdout:
        await ws_send({"type": "log", "line": line.decode()})
    await proc.wait()
```

### 11.2 Result Loading

After any analysis run, the backend reads the output CSV files produced by the existing scripts. These are well-defined:
- `raw_comparison.csv` from `test_solvers.py`
- `dense_trajectory_*.csv` from `feasibility_analysis.py`
- Per-trajectory metric arrays

These are serialized to JSON and sent to the frontend for plotting. The frontend never reads files directly.

---

## 12. Feasibility Analysis Mode

> Read `feasibility_analysis.py` fully before implementing. Understand `FeasibilityConfig`, `process_toolpath()`, and all its output phases.

The feasibility pipeline runs: `IK → TOPP-RA → C1 velocity → C0 continuity → singularity → manipulability → task-space velocity`

### 12.1 What the UI Exposes

Each feasibility check has a toggle in the Config Panel:
- Singularity: enabled toggle, threshold, mode (unified/classified)
- Manipulability: enabled toggle, warning threshold
- C0 Continuity: enabled toggle, max jump threshold
- C1 Continuity: enabled toggle (requires TOPP-RA)
- TOPP-RA: always runs if IK succeeds (no enabled toggle)
- Task-space velocity: enabled toggle, speed limit

These toggles map directly to the `FeasibilityConfig` dataclass fields. Read `utils/config_loader.py` for the exact field names.

### 12.2 Time-Parameterized Trajectory

After TOPP-RA runs, the user has the option to generate a dense time-parameterized trajectory. This is `ToppraResult` from `core/topp_check.py` containing `q(t)`, `qdot(t)`, `qddot(t)` sampled at regular time intervals. When this is available:
- The timeline can show real time (seconds) instead of waypoint index
- The TOPP-RA plot group shows these profiles
- The 3D animation plays at the correct speed

---

## 13. Plot System — Groups, Timeline Cursors, ECFX Branches

### 13.1 The Timeline Cursor — Core Mechanism

Every plot in the application has a synchronized vertical cursor line. When the timeline index changes, ALL plot cursors update to the same index simultaneously. The robot in the 3D Viser viewer also updates to show that waypoint's configuration.

**How it works technically:**

```typescript
// usePlotCursors.ts
import { useCallback } from 'react'
import Plotly from 'plotly.js'

// Each plot is registered with a unique div id
const plotRegistry = new Map<string, string>() // plotId → divId

export function broadcastCursorUpdate(waypointIndex: number, timeValue?: number) {
  // For each registered plot:
  for (const [plotId, divId] of plotRegistry) {
    const xValue = plotId.startsWith('topp_') ? timeValue : waypointIndex
    Plotly.relayout(divId, {
      'shapes[0].x0': xValue,
      'shapes[0].x1': xValue,
    })
    // This is a shape annotation update — NOT a data redraw.
    // Plotly.relayout with shapes is sub-millisecond.
  }
}
```

Every plot must be initialized with a vertical shape at x=0:

```typescript
const initialLayout = {
  shapes: [{
    type: 'line',
    x0: 0, x1: 0,
    y0: 0, y1: 1,
    yref: 'paper',
    line: { color: '#ef4444', width: 2, dash: 'solid' }
  }],
  // ... rest of layout
}
```

**Critical rule:** Never call `Plotly.react()` or `Plotly.newPlot()` on a cursor update. Only call `Plotly.relayout()` with the shape coordinate. This is what keeps scrubbing instantaneous even with 20+ plots visible.

### 13.2 Plot Groups

Plots are organized into collapsible groups in the right panel. All plots within a group share a linked x-axis (pan/zoom one → all follow):

```
Group: Kinematics
  ├── Joint Angles (J1–J6, 6 lines over waypoint index)
  └── TCP Position (x, y, z over waypoint index)

Group: Feasibility — Singularity
  ├── Manipulability Index (with warning threshold band)
  ├── Condition Number (log scale, with threshold line)
  ├── Minimum Singular Value (with threshold line)
  └── Singular Value Spectrum (heatmap: waypoints × 6 SVs)

Group: Feasibility — Continuity
  ├── C0 Joint Jump (6 lines per joint + max line)
  └── C1 Velocity Fraction (6 lines: TOPP-RA velocity / limit)

Group: TOPP-RA  (x-axis is real time in seconds, not waypoint index)
  ├── Joint Positions vs Time
  ├── Joint Velocities vs Time
  ├── Joint Accelerations vs Time
  └── Task-Space TCP Speed (mm/s vs time)

Group: ECFX Branches  (hidden by default, open explicitly — see Section 14)
  ├── J1 — All ECFX Branches
  ├── J2 — All ECFX Branches
  ├── J3 — All ECFX Branches
  ├── J4 — All ECFX Branches
  ├── J5 — All ECFX Branches
  ├── J6 — All ECFX Branches
  ├── Active Branch Selector (overlay on each subplot)
  └── Branch Count + LS/Exact Timeline

Group: Solver Comparison  (visible only when comparison mode active)
  ├── FK Position Error (Pinocchio vs EAIK vs reference)
  ├── FK Per-Axis Deltas
  ├── IK Joint Deltas (Δj1..Δj6)
  └── IK Success Rate
```

### 13.3 Group Rendering Rules

- Each group is a `<PlotGroup>` component with a collapsible header
- Groups that have no data are hidden entirely (not collapsed — hidden)
- ECFX Branch group is hidden by default even when data is present; user must explicitly open it
- Within a group, all plots are stacked vertically and share the same x-axis range
- Plotly's `syncviews` or manual `on_relayout` event must link zoom/pan across all plots in the same group
- A "Reset Zoom" button per group resets all plots in that group to the full data range

### 13.4 Trajectory Selector

When multiple sub-trajectories exist, a set of tabs or a dropdown at the top of the plot dashboard lets the user switch which trajectory's data is being plotted. Switching:
1. Replaces all plot data with the selected trajectory's data
2. Resets the timeline to index 0
3. Updates the 3D scene to show only the selected trajectory

---

## 14. ECFX Branch Classification (EAIK Solutions)

> This is a significant update from how EAIK solutions were previously treated. Read `core/eaik_ik_solver.py` to understand `compute_ecfx()` and `ECFXLabel`.

### 14.1 What ECFX Is

EAIK analytical IK returns multiple solutions per waypoint. Previously, these were treated as randomly indexed branches (0, 1, 2...). **They are now classified using ECFX labels** — the equivalent of ABB RobotStudio's `confdata`.

```python
# From core/eaik_ik_solver.py (do not copy — import it)
class ECFXLabel(NamedTuple):
    cf1: int   # floor(joint1_deg / 90) — quadrant of axis 1
    cf4: int   # floor(joint4_deg / 90) — quadrant of axis 4
    cf6: int   # floor(joint6_deg / 90) — quadrant of axis 6
    cfx: int = 0  # always 0 for IRB 1300

def compute_ecfx(q_rad: np.ndarray) -> ECFXLabel:
    """Must be called on raw EAIK angle BEFORE normalization."""
    q_deg = np.degrees(q_rad)
    return ECFXLabel(
        cf1=int(math.floor(q_deg[0] / 90.0)),
        cf4=int(math.floor(q_deg[3] / 90.0)),
        cf6=int(math.floor(q_deg[5] / 90.0)),
    )
```

Each solution branch now has a human-readable identity: `cf1=0, cf4=-1, cf6=1` for example, instead of "Branch 2". This matches what RobotStudio would display.

### 14.2 ECFX Data Structure Per Waypoint

For each waypoint, the backend computes and stores:

```python
@dataclass
class WaypointECFXData:
    waypoint_index: int
    solutions: List[SolutionEntry]
    selected_index: int           # which solution the pipeline chose

@dataclass
class SolutionEntry:
    branch_index: int             # 0–7 raw index from EAIK
    ecfx: ECFXLabel               # cf1, cf4, cf6, cfx
    joint_angles_deg: List[float] # 6 values
    is_ls: bool                   # True = least-squares (approximate)
    fk_error_mm: float            # FK error of this solution vs target TCP
```

### 14.3 ECFX Branch Plots — Detailed Specification

The ECFX Branch plot group is **hidden by default**. The user opens it via an "ECFX Details" toggle button. It contains **three dedicated graphs** (not 6 joint subplots — focus on the three cf-axes):

#### Graph 1: cf1 Evolution (Joint 1 quadrant)
- X-axis: waypoint index
- Y-axis: joint 1 angle in degrees
- Each solution at each waypoint is a point, colored by its `cf1` value
- Color mapping: `cf1 = -2` → blue, `-1` → cyan, `0` → green, `1` → orange, `2` → red
- Exact solutions: filled circle marker, size 6
- LS solutions: hollow circle marker, same color but 50% opacity
- The **selected solution** at each waypoint: larger marker with black border (higher z-index, always visible above other points)
- If waypoints have many solutions overlapping (high density), the selected solution must visually dominate — use `marker.line.width = 2` on selected, `0` on others

#### Graph 2: cf4 Evolution (Joint 4 quadrant)
- Same structure as cf1 graph but for joint 4 angles and `cf4` value color
- Shows wrist configuration variation

#### Graph 3: cf6 Evolution (Joint 6 quadrant)
- Same structure but for joint 6 angles and `cf6` value color
- Shows wrist-flip variation; large divergence between cf6 groups is typical

#### Below the three graphs: ECFX Summary Panel
- A small table per waypoint: columns are the unique ECFX labels present (e.g., `(0,-1,1)`, `(1,-1,0)`, `(0,0,1)` ...)
- Cells show how many solutions have each ECFX label at each waypoint
- Color-coded background per cell matching the cf color scheme
- Selected solution highlighted with a border

#### Important Rendering Note for High Waypoint Count

When a trajectory has many waypoints (>200), showing all solution scatter points becomes visually cluttered. Apply the following:

```
if num_waypoints > 200:
    - Show selected solution points at full opacity (z-index front)
    - Show non-selected solution points at 20% opacity
    - Cluster nearby solutions using point binning (no subsampling — all data is kept,
      just rendered at lower opacity)
    - Show "Solution count" as a heatmap background color on the x-axis strip
      rather than individual points for non-selected solutions
```

The selected solution (the one the pipeline actually used) must always be fully visible at any zoom level and any trajectory length.

### 14.4 Timeline Cursor in ECFX Plots

Same cursor mechanism as all other plots. When the timeline scrubs to waypoint i:
- Cursor moves in all three ECFX graphs
- The ECFX Summary Panel highlights the row for waypoint i
- The 3D robot shows the selected solution's joint configuration

### 14.5 ECFX in Solver Comparison Mode

When comparing Pinocchio vs EAIK:
- EAIK solutions are colored by ECFX label (cf1/cf4/cf6)
- Pinocchio has only one solution per waypoint — shown as a single line in neutral grey
- A separate "ECFX Consistency" metric plot shows whether the selected solution's ECFX label stays consistent across the trajectory (jumps indicate configuration changes)

---

## 15. 3D Visualization — Viser Embedding

> Read `visualize_robot.py` in the existing project. This is the reference implementation of a Viser viewer. Understand how it loads URDFs and uses joint sliders before building `viser_server.py`.

### 15.1 How Viser Embedding Works

Viser runs a web server and serves its own React-based 3D client. The Live Visualizer **embeds this client inside an `<iframe>`**.

```typescript
// ViserEmbed.tsx
export function ViserEmbed() {
  return (
    <iframe
      src="http://localhost:8081"    // Viser server port
      className="w-full h-full border-0"
      title="3D Robot Viewer"
    />
  )
}
```

For embedded/static scenes (e.g., exporting a trajectory preview), use the `?playbackPath=` parameter:

```
http://localhost:8081/viser-client/?playbackPath=http://localhost:8081/recordings/trajectory.viser
```

To create a `.viser` recording file from a computed trajectory:

```python
# In viser_server.py
serializer = server.get_scene_serializer()

for i, q in enumerate(joint_trajectory):
    urdf_vis.update_cfg(q)
    serializer.insert_sleep(1.0 / 30.0)  # 30fps playback

Path("recordings/trajectory.viser").write_bytes(serializer.serialize())
```

### 15.2 Viser Server Responsibilities

`viser_server.py` manages:

- Loading and displaying the selected robot URDF using `viser.extras.ViserUrdf`
- Drawing trajectory paths as `server.scene.add_spline_catmull_rom()`
- Showing waypoint cloud as `server.scene.add_point_cloud()` (colored by status)
- Updating robot joint config on timeline scrub via `urdf_vis.update_cfg(q)`
- Drawing coordinate frames (base, knife, TCP) via `server.scene.add_frame()`
- Drawing ECFX ghost robots (semi-transparent poses for all solutions at selected waypoint)
- Receiving update commands from the FastAPI server via an internal asyncio queue

### 15.3 Scene Update Protocol

FastAPI server → Viser server communication uses a Python `asyncio.Queue`:

```python
# scene_state.py
from asyncio import Queue
scene_update_queue: Queue = Queue()

# SceneUpdateCommand types:
# { "cmd": "load_robot", "urdf_path": str, "robot_name": str }
# { "cmd": "draw_trajectory", "waypoints": List[List[float]], "colors": List[str] }
# { "cmd": "set_waypoint", "index": int, "q": List[float] }
# { "cmd": "draw_frame", "name": str, "pos": List[float], "wxyz": List[float] }
# { "cmd": "clear_scene" }
# { "cmd": "show_ecfx_ghosts", "solutions": List[Dict] }
```

### 15.4 Waypoint Color Coding

| Color | Hex | Condition |
|-------|-----|-----------|
| Green | `#22c55e` | Reachable, non-singular, within all limits |
| Yellow | `#eab308` | Manipulability < warning threshold |
| Orange | `#f97316` | C0 joint jump violation |
| Red | `#ef4444` | IK failed (no solution or joint limits) |
| Blue | `#3b82f6` | Currently selected by timeline cursor |
| Purple | `#a855f7` | LS (least-squares) solution — approximate |
| Grey | `#6b7280` | RobotStudio reference waypoints |

---

## 16. Config as UI Items

> Read `utils/config_loader.py` fully. The `FeasibilityConfig` dataclass is the source of truth.

### 16.1 Philosophy

No YAML file editing. Every config parameter that was previously in `batch_feasibility_config.yaml` is exposed as a UI element. The backend reads these from the frontend request, not from files.

### 16.2 UI Mapping

```
FeasibilityConfig field               → UI Component
─────────────────────────────────────────────────────
solver: "pin" | "eaik"               → Radio buttons: "Pinocchio" / "EAIK" / "Both"
reachability.generate_graphs         → (always true in visualizer — not exposed)
singularity.enabled                  → Toggle switch
singularity.mode                     → Radio: "unified" / "classified"
singularity.threshold                → Number input with default + reset
singularity.j5_threshold_deg         → Number input
manipulability.enabled               → Toggle switch
manipulability.warning               → Number input
manipulability.translational_warning → Number input
continuity.enabled                   → Toggle switch
continuity.safety_factor             → Number input + slider (0.5 → 2.0)
continuity.default_speed_mm_s        → Number input
time_parameterization.enabled        → Toggle switch
time_parameterization.max_gap_mm     → Number input
time_parameterization.interpolate_sparse → Toggle switch
topp_ra.generate_graphs              → (always true in visualizer — not exposed)
eaik_multi_solution.enabled          → Toggle switch (EAIK only)
eaik_multi_solution.weights.c0       → Number input (EAIK only)
eaik_multi_solution.weights.singularity → Number input (EAIK only)
max_ik_failures_per_trajectory       → Number input
```

### 16.3 Config Panel Layout

The Config Panel is a collapsible sidebar section with groups matching the YAML structure:

```
▼ Solver & Robot
    [Pinocchio] [EAIK] [Both]
    EE Frame: [Link_6 ▾]
    Max IK failures: [1]

▼ Singularity
    [✓ Enabled]  Mode: [unified ▾]
    Threshold: [0.01]  J5 threshold: [0.76°]

▼ Manipulability
    [✓ Enabled]
    Warning: [0.001]  Translational: [0.001]

▼ Continuity
    [✓ Enabled]
    Safety factor: [1.05 ──●──────]  Speed: [100 mm/s]

▼ TOPP-RA
    Always runs when IK succeeds
    Max gap: [5.0 mm]  [✓ Interpolate sparse]

▼ EAIK Multi-Solution  (visible only when EAIK selected)
    [✓ Enabled]
    C0 weight: [10.0]  Singularity: [1.0]  Manip: [0.5]
```

All values have a "Reset to default" button (circular arrow icon). Changes take effect on the next Run — they do not re-run automatically.

---

## 17. TeleOp Tab

> TeleOp lives on a separate tab (`/teleop`). It shares the same Python backend process but is a completely separate UI route.

### 17.1 Overview

TeleOp allows the user to drive the robot in real time from the browser in two modes:

| Mode | How It Works |
|------|-------------|
| **Task Space** | Keyboard keys increment/decrement TCP position (X/Y/Z) and orientation (roll/pitch/yaw). IK is solved in real time. |
| **Joint Space** | Keys 1–6 select a joint. Arrow keys increment/decrement that joint's angle. FK is computed after each change. |

Both modes support recording waypoints and exporting trajectories in all formats compatible with the existing pipeline.

### 17.2 WebSocket TeleOp Protocol

TeleOp uses a persistent WebSocket connection at `/ws/teleop`. All keyboard events are sent as JSON messages:

```typescript
// Frontend sends:
{ "type": "key", "action": "translate_x_plus", "mode": "task_space" }
{ "type": "key", "action": "joint_plus", "joint": 2, "mode": "joint_space" }
{ "type": "record_waypoint" }
{ "type": "start_recording" }
{ "type": "stop_recording" }
{ "type": "set_mode", "mode": "task_space" | "joint_space" }

// Backend responds:
{ "type": "robot_state", "q": [j1,...,j6], "tcp": [x,y,z,qw,qx,qy,qz] }
{ "type": "metrics", "manipulability": 0.043, "condition_number": 12.1,
  "min_singular_value": 0.002, "ik_method": "converged",
  "joint_limit_fractions": [0.8, 0.3, 0.9, 0.5, 0.1, 0.7] }
{ "type": "ik_failure", "reason": "no_solutions" }
{ "type": "waypoint_recorded", "index": 12, "total": 13 }
```

### 17.3 Real-Time IK Loop (Backend)

```python
# teleop/teleop_handler.py
async def handle_teleop_key(action: str, current_state: TeleopState) -> TeleopState:
    if current_state.mode == "task_space":
        # Apply delta to current TCP pose
        target_pos, target_quat = apply_delta(action, current_state.tcp_pos, current_state.tcp_quat)
        # Run IK
        success, q, info = ik_solver.solve_with_retries(target_pos, target_quat, q_init=current_state.q)
        if success:
            current_state.q = q
            fk_result = fk_solver.solve(q)
            current_state.tcp_pos = fk_result.position
            current_state.tcp_quat = fk_result.quaternion
        # Compute live metrics
        J = fk_solver.get_jacobian(current_state.q)
        metrics = compute_live_metrics(J)
        
    elif current_state.mode == "joint_space":
        joint_idx = action_to_joint_index(action)
        delta = action_to_delta(action)
        q_new = current_state.q.copy()
        q_new[joint_idx] = clamp(q_new[joint_idx] + delta, joint_limits[joint_idx])
        current_state.q = q_new
        fk_result = fk_solver.solve(q_new)
        current_state.tcp_pos = fk_result.position
        current_state.tcp_quat = fk_result.quaternion
        J = fk_solver.get_jacobian(current_state.q)
        metrics = compute_live_metrics(J)
    
    # Update Viser scene
    await scene_update_queue.put({"cmd": "set_joint_config", "q": current_state.q.tolist()})
    return current_state, metrics
```

### 17.4 Keyboard Mapping

Full keyboard map (user-remappable from settings):

| Key | Task Space | Joint Space |
|-----|-----------|-------------|
| W / S | TCP +Z / −Z | Active joint +step / −step |
| A / D | TCP −X / +X | Previous / Next joint |
| Q / E | TCP +Y / −Y | Active joint large step |
| I / K | Rotate +pitch / −pitch | Joint 1 ±step |
| J / L | Rotate +yaw / −yaw | Joint 2 ±step |
| U / O | Rotate +roll / −roll | Joint 3 ±step |
| 1–6 | (none) | Select joint 1–6 |
| Space | Record waypoint | Record waypoint |
| R | Start/stop auto-record | Start/stop auto-record |
| Backspace | Delete last waypoint | Delete last waypoint |
| Enter | Save trajectory | Save trajectory |
| Escape | Clear trajectory | Clear trajectory |
| Tab | Switch to joint space | Switch to task space |

### 17.5 Live Metrics HUD

The right panel in TeleOp mode shows a persistent HUD that updates after every keypress:

- **Manipulability gauge**: large colored number + radial bar (green/yellow/red)
- **Condition number**: horizontal bar on log scale
- **Min singular value**: draining bar towards zero
- **Joint limit fractions**: 6 small bars (J1–J6), each showing `current_angle / joint_range`; red when < 10% margin
- **IK method badge**: `EAIK: converged` (green) / `Pin: random` (yellow) / `FAILED` (red)
- **TCP position**: `x: 0.342m  y: -0.128m  z: 0.456m`
- **TCP orientation**: roll/pitch/yaw in degrees
- **Recording indicator**: red `● REC` + waypoint count + elapsed time (blinks during auto-record)

### 17.6 Recording & Export

After stopping a recording, a review panel appears:

- Playback: play the recorded trajectory on the robot (uses timeline, same as analysis mode)
- Trim: set start/end waypoint index
- Resample: choose target spacing in mm
- Export formats:

| Format | Description |
|--------|-------------|
| Toolpath CSV (T_P_K) | TCP poses inverse-transformed back through knife pose. Compatible with `feasibility_analysis.py` |
| Base frame CSV (T_B_P) | TCP poses in base frame. Use with `--base_frame` flag |
| RobotStudio validation CSV | Joint angles + TCP in RS format. Compatible with `test_solvers.py` |
| Joint space CSV | 6 joint angles per timestep in degrees |

Saved files go to `Assets/Robot APCC/Toolpaths/TeleOp_Recordings/`.

After saving, a "Run Feasibility on This Trajectory" button appears. Clicking it takes the user to the Analysis tab with this file pre-loaded.

---

## 18. Backend Python Server

### 18.1 FastAPI Setup

```python
# backend/server.py
from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Run with:
# uvicorn visualizer.backend.server:app --port 8080 --reload
```

### 18.2 Session Management

Each user session (browser tab) gets a unique session ID. All uploaded files, detection results, transform results, and analysis outputs are stored under `output/visualizer_sessions/{session_id}/`. Sessions expire after 24 hours.

### 18.3 Key Endpoints

```
POST /api/upload
  Body: multipart/form-data, file: CSV
  Returns: { session_id, num_rows, raw_columns, preview_rows }

POST /api/detect/{session_id}
  Body: { column_map?: Dict[str, str] }
  Returns: { has_task_space, has_joint_space, trajectories, detected_columns, unknown_columns }

POST /api/configure/{session_id}
  Body: { use_base_frame, knife_name?, robot_name }
  Returns: { transformed_waypoints_preview, robot_metadata, joint_limits }

POST /api/run/{session_id}
  Body: { mode, config: FeasibilityConfigJSON }
  Returns: { job_id }
  (Actual results streamed via WebSocket)

GET /api/results/{session_id}/{job_id}
  Returns: { status, plots_data, trajectory_data }

WS /ws/stream/{session_id}/{job_id}
  Streams: { type: "log" | "progress" | "partial_result" | "done" | "error" }

WS /ws/teleop/{session_id}
  Bidirectional: key events in, robot state + metrics out
```

### 18.4 Startup

Both servers start together:

```python
# start.py
import subprocess, sys

viser_proc = subprocess.Popen([sys.executable, "visualizer/backend/viser_server.py"])
uvicorn_proc = subprocess.Popen([
    sys.executable, "-m", "uvicorn",
    "visualizer.backend.server:app",
    "--port", "8080"
])
# Or use a process manager like supervisord
```

---

## 19. API Contract — Frontend ↔ Backend

All API responses follow this envelope:

```typescript
interface APIResponse<T> {
  ok: boolean
  data?: T
  error?: string
}
```

### Key TypeScript Types

```typescript
// types/data.ts

interface DetectionResult {
  has_task_space: boolean
  has_joint_space: boolean
  num_trajectories: number
  num_waypoints_per_trajectory: number[]
  detected_columns: Record<string, string>  // column_name → role
  unknown_columns: string[]
}

interface RobotOption {
  name: string
  urdf_path: string
  reach_mm: number
  joint_limits: [number, number][]  // [min_deg, max_deg] per joint
  velocity_limits: number[]          // deg/s per joint
}

interface KnifeOption {
  name: string
  translation_mm: [number, number, number]
  quaternion: [number, number, number, number]  // w, x, y, z
}

interface WaypointStatus {
  index: number
  reachable: boolean
  manipulability: number
  condition_number: number
  c0_violation: boolean
  c1_violation: boolean
  ecfx?: ECFXLabel          // only when EAIK
  ik_method: string
}

interface ECFXLabel {
  cf1: number
  cf4: number
  cf6: number
  cfx: number
}

interface SolutionBranch {
  branch_index: number
  ecfx: ECFXLabel
  joint_angles_deg: number[]
  is_ls: boolean
  fk_error_mm: number
}

interface PlotData {
  group: string
  plot_id: string
  title: string
  x: number[]
  series: PlotSeries[]
  x_label: string
  y_label: string
}

interface PlotSeries {
  name: string
  y: number[]
  color?: string
  dash?: 'solid' | 'dash' | 'dot'
  mode?: 'lines' | 'markers' | 'lines+markers'
  opacity?: number
}

interface TeleopRobotState {
  q: number[]                    // 6 joint angles in radians
  tcp: [number, number, number, number, number, number, number]  // x,y,z,qw,qx,qy,qz
  metrics: TeleopMetrics
}

interface TeleopMetrics {
  manipulability: number
  condition_number: number
  min_singular_value: number
  joint_limit_fractions: number[]  // 0–1 for each joint
  ik_method: string
}
```

---

## 20. Implementation Phases

Build in this order. Each phase produces a working, demonstrable result.

### Phase 1 — Foundation (Week 1)
Goal: App shell loads, Viser iframe works, robot can be loaded.

- [ ] Vite + React + TypeScript + Tailwind project setup
- [ ] App shell with Analysis/TeleOp tab routing
- [ ] FastAPI server with CORS and basic health endpoint
- [ ] Viser server: loads URDF, shows robot at home pose
- [ ] ViserEmbed component: iframe pointing to Viser port
- [ ] Robot selector dropdown populated from `robots_config.yaml`
- [ ] Basic scene update: selecting robot in frontend → Viser shows it

### Phase 2 — CSV Upload & Detection (Week 1–2)
Goal: User can upload a CSV and see what data is detected.

- [ ] UploadStep component with react-dropzone
- [ ] POST /api/upload endpoint
- [ ] data_detection.py module (read csv_loader scripts first)
- [ ] DetectStep component showing detected columns
- [ ] Column mapper UI for unknown columns
- [ ] Frame configuration step (base frame vs knife, knife selector from knife_config.yaml)
- [ ] Transform endpoint calling `transform_trajectories_to_base_frame()`
- [ ] Robot selection step
- [ ] Transformed waypoints drawn in Viser as a point cloud preview

### Phase 3 — IK/FK Runs & Basic Plots (Week 2–3)
Goal: User can run IK and see joint trajectories plotted with timeline cursor.

- [ ] Action selection step
- [ ] ConfigPanel with key config toggles mapped to FeasibilityConfig
- [ ] RunPanel with Run button + WebSocket log streaming
- [ ] Run IK endpoint: calls `create_solvers()` + `ik_solver.solve_with_retries()` per waypoint
- [ ] Run FK endpoint: calls `fk_solver.solve()` per waypoint
- [ ] PlotGroup + TimelinePlot components with Plotly
- [ ] Timeline cursor mechanism (shape annotation, broadcast on index change)
- [ ] Kinematics group plots (joint angles + TCP position)
- [ ] TimelineBar component (scrubber, play/pause, step, jump-to-index)
- [ ] Viser: robot animates on timeline scrub (urdf_vis.update_cfg on each index)
- [ ] Waypoint color coding by reachability status

### Phase 4 — Feasibility & Full Plot Groups (Week 3–4)
Goal: Full feasibility pipeline runs, all plot groups visible.

- [ ] Feasibility run endpoint calling `process_toolpath()` via subprocess
- [ ] Result loading from dense_trajectory CSV output
- [ ] Singularity, manipulability, continuity plot groups
- [ ] TOPP-RA group with real-time axis
- [ ] Waypoint color coding upgraded (all conditions: singular, C0, C1 violations)
- [ ] Multi-trajectory support (trajectory selector tabs, switching trajectory updates all plots + scene)

### Phase 5 — ECFX Branch Plots (Week 4)
Goal: Full ECFX-colored branch visualization.

- [ ] Backend: compute `compute_ecfx()` for all EAIK solutions, store per waypoint
- [ ] EcfxBranchPlot component: three graphs (cf1, cf4, cf6 evolution)
- [ ] ECFX color mapping + scatter point rendering
- [ ] Selected solution higher z-index / always visible
- [ ] High density rendering (opacity reduction for non-selected)
- [ ] ECFX Summary Panel (table view)
- [ ] ECFX ghost robots in Viser for selected waypoint
- [ ] Group hidden by default, opened via toggle button

### Phase 6 — TeleOp Tab (Week 4–5)
Goal: User can drive the robot with keyboard in both modes.

- [ ] TeleOp WebSocket endpoint `/ws/teleop`
- [ ] teleop_handler.py with task-space and joint-space IK/FK loops
- [ ] TeleopPanel + KeyboardMap components
- [ ] HudMetrics component updating at 30fps from WS
- [ ] Keyboard event capture in browser → WS send
- [ ] Viser updates robot on every WS response
- [ ] Manual waypoint recording (Space key)
- [ ] Auto-record mode (R key)
- [ ] Trajectory review panel
- [ ] Export in all 4 CSV formats

### Phase 7 — Polish & Solver Comparison (Week 5–6)
Goal: FK/IK comparison mode, UX polish.

- [ ] Comparison mode: run both solvers, overlay paths in Viser
- [ ] Solver comparison plot group (FK error, IK deltas)
- [ ] ECFX consistency plot in comparison mode
- [ ] RobotStudio reference path overlay (3rd spline in grey)
- [ ] Config panel: all FeasibilityConfig fields exposed
- [ ] Resizable panel layout (react-resizable-panels)
- [ ] Session persistence (reload page → session still loaded)
- [ ] Error handling throughout (network errors, IK failures, malformed CSV)

---

## 21. Important Constraints & Rules

### Never Do

- ❌ Modify any file in the existing `Robotics-APCC/` project
- ❌ Reimplement FK, IK, singularity, manipulability, TOPP-RA, or transform logic
- ❌ Generate static PNG files — all plots are live Plotly charts
- ❌ Use `Plotly.react()` or `Plotly.newPlot()` on timeline scrub (only `Plotly.relayout()` for cursor updates)
- ❌ Read YAML config files in the frontend — all config comes from API
- ❌ Use `localStorage` for session state (use server-side sessions)
- ❌ Block the Python event loop during IK computation (use `asyncio.run_in_executor` or subprocess)

### Always Do

- ✅ Read the relevant existing script before building any feature that touches it
- ✅ Call `compute_ecfx()` on raw EAIK angles BEFORE any normalization
- ✅ Show selected ECFX solution with higher visual priority (z-index, larger marker, full opacity)
- ✅ Keep Viser, FastAPI, and frontend as three separately started processes
- ✅ Stream stdout to the frontend log panel for all long-running analysis jobs
- ✅ Make all config toggles produce visible changes in the run — map them to `FeasibilityConfig` fields exactly
- ✅ Keep the 3D viewport and the timeline cursor always in sync — same index, always
- ✅ Expose robot selector from `robots_config.yaml` and knife selector from `knife_config.yaml` — never hardcode names

### ECFX-Specific Rules

- ✅ Color by `cf1`/`cf4`/`cf6` value, not by branch index (index is arbitrary; ECFX is meaningful)
- ✅ Show the selected solution (the one the pipeline chose) at full opacity and higher z-index always
- ✅ Distinguish LS solutions (hollow/dashed) from exact solutions (filled/solid)
- ✅ Hide the ECFX group by default — only show on explicit user request

### Performance Rules

- The timeline cursor update must complete in < 16ms (one frame at 60fps)
- The TeleOp IK loop must respond to each keypress in < 50ms end-to-end
- Do not re-render entire Plotly figures on cursor update — shapes only
- Pre-compute all IK/FK results before allowing timeline scrubbing (no on-demand computation during scrub)

---

*This README is written for a coding agent. Start with Section 2 (read existing files), then proceed through sections in order. Do not skip pre-reading — the existing scripts define the data contracts that everything else depends on.*
