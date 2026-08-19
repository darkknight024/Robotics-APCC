#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import csv
from collections import defaultdict
from typing import Any, Optional, Dict, List, Tuple

import numpy as np

# --- 1. Auto-install Dependencies ---
# List of (import_name, pip_install_name). Use same name twice if they match.
required_packages = [
    ("viser", "viser"),
    ("yourdfpy", "yourdfpy"),
    ("scipy", "scipy"),
    ("imageio", "imageio"),
    ("yaml", "pyyaml"),  # PyPI package is pyyaml, module is yaml
]
for import_name, pip_name in required_packages:
    try:
        __import__(import_name)
    except ImportError:
        print(f"{import_name} not found. Installing {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        except Exception as e:
            print(f"Failed to install {pip_name}: {e}")
            sys.exit(1)

import viser
from viser.extras import ViserUrdf
import yourdfpy
import yaml
from scipy.spatial.transform import Rotation
import imageio.v3 as iio

def load_config(config_path="config/robots_config.yaml"):
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _T_to_pos_wxyz(T: np.ndarray):
    pos = T[:3, 3]
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
    return pos, wxyz


def _load_trimesh(path: str, scale: float = 1.0):
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if abs(scale - 1.0) > 1e-12:
        mesh.apply_scale(scale)
    return mesh


def add_collision_scene_meshes(server: viser.ViserServer, scene_yaml: str, repo_root: str) -> int:
    """Add static cell STLs from collision_objects.yaml (position_mm + quat_wxyz)."""
    if not os.path.exists(scene_yaml):
        print(f"Scene YAML not found: {scene_yaml}")
        return 0
    with open(scene_yaml, "r") as f:
        doc = yaml.safe_load(f) or {}
    n = 0
    for obj in doc.get("objects") or []:
        mesh_rel = obj.get("mesh_path")
        if not mesh_rel:
            continue
        mesh_path = mesh_rel if os.path.isabs(mesh_rel) else os.path.join(repo_root, mesh_rel)
        if not os.path.exists(mesh_path):
            print(f"  skip missing mesh: {mesh_path}")
            continue
        pos_mm = obj.get("position_mm") or [0, 0, 0]
        quat = obj.get("quat_wxyz") or obj.get("quaternion") or [1, 0, 0, 0]
        scale = float(obj.get("scale", 1.0))
        try:
            mesh = _load_trimesh(mesh_path, scale=scale)
        except Exception as e:
            print(f"  failed to load {mesh_path}: {e}")
            continue
        pos_m = np.array(pos_mm, dtype=float) * 0.001
        wxyz = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        name = str(obj.get("name", f"scene_{n}"))
        server.scene.add_mesh_trimesh(f"/scene/{name}", mesh, position=pos_m, wxyz=wxyz)
        n += 1
    print(f"Loaded {n} scene STL(s) from {scene_yaml}")
    return n


def _fixture_config_path(repo_root: str) -> str:
    return os.path.join(repo_root, "config", "fixture_config.yaml")


def list_fixture_names(repo_root: str) -> List[str]:
    """Return fixture names from fixture_config.yaml that have a non-empty stl."""
    cfg_path = _fixture_config_path(repo_root)
    if not os.path.exists(cfg_path):
        return []
    with open(cfg_path, "r") as f:
        doc = yaml.safe_load(f) or {}
    names: List[str] = []
    for name, data in (doc.get("fixtures") or {}).items():
        if data and str(data.get("stl") or "").strip():
            names.append(str(name))
    return names


def add_fixture_mesh(server: viser.ViserServer, fixture_name: str, repo_root: str):
    """Load one fixture STL. Returns (handle, T_link_fixture, parent_link) or (None, None, None)."""
    cfg_path = _fixture_config_path(repo_root)
    if not os.path.exists(cfg_path):
        return None, None, None
    with open(cfg_path, "r") as f:
        doc = yaml.safe_load(f) or {}
    data = (doc.get("fixtures") or {}).get(fixture_name)
    stl_rel = str((data or {}).get("stl") or "").strip()
    if not data or not stl_rel:
        return None, None, None
    stl_path = stl_rel if os.path.isabs(stl_rel) else os.path.join(repo_root, stl_rel)
    if not os.path.exists(stl_path):
        print(f"Fixture STL not found: {stl_path}")
        return None, None, None
    origin = data.get("origin") or {}
    xyz = list(origin.get("xyz") or [0, 0, 0])
    rpy = list(origin.get("rpy") or [0, 0, 0])
    from utils.urdf_loader import build_transform_from_xyz_rpy
    T = build_transform_from_xyz_rpy(xyz, rpy)
    scale = float(data.get("scale", 1.0))
    try:
        mesh = _load_trimesh(stl_path, scale=scale)
    except Exception as e:
        print(f"Failed to load fixture STL {stl_path}: {e}")
        return None, None, None
    parent_link = str(data.get("parent_link") or "Link_6")
    handle = server.scene.add_mesh_trimesh(f"/fixture/{fixture_name}", mesh)
    print(f"Loaded fixture '{fixture_name}' on {parent_link}: {stl_path}")
    return handle, T, parent_link


def add_fixture_meshes(
    server: viser.ViserServer,
    fixture_names: List[str],
    repo_root: str,
) -> List[Tuple[Any, np.ndarray, str]]:
    """Load every named fixture that has an STL. Returns list of (handle, T, parent_link)."""
    loaded: List[Tuple[Any, np.ndarray, str]] = []
    for name in fixture_names:
        handle, T, parent = add_fixture_mesh(server, name, repo_root)
        if handle is not None and T is not None and parent is not None:
            loaded.append((handle, T, parent))
    return loaded

def update_frames_and_labels(server: viser.ViserServer, urdf_model: yourdfpy.URDF, handles: dict, visible: bool):
    """Updates the position and visibility of debug frames and labels."""
    if not visible:
        for h_list in handles.values():
            for h in h_list:
                h.visible = False
        return

    # Get all link transforms relative to world
    # yourdfpy manages the scene graph
    scene = urdf_model.scene
    
    # Iterate over all links
    for link_name in urdf_model.link_map.keys():
        # Get transform of this link
        transform = scene.graph.get(link_name)[0]
        
        # Extract Position
        pos = transform[:3, 3]
        # Extract Rotation (matrix to quaternion)
        rot_matrix = transform[:3, :3]
        quat = Rotation.from_matrix(rot_matrix).as_quat() # x, y, z, w
        # Viser expects w, x, y, z
        wxyz = (quat[3], quat[0], quat[1], quat[2])

        # Create handles if they don't exist
        if link_name not in handles:
            # Add Frame
            frame_handle = server.scene.add_frame(
                f"/frames/{link_name}",
                show_axes=True,
                axes_length=0.15,
                axes_radius=0.005,
                position=pos,
                wxyz=wxyz,
            )
            
            # Add Label
            label_handle = server.scene.add_label(
                f"/labels/{link_name}",
                text=link_name,
                position=pos,
            )
            
            handles[link_name] = [frame_handle, label_handle]
        else:
            # Update existing
            frame_handle, label_handle = handles[link_name]
            
            frame_handle.position = pos
            frame_handle.wxyz = wxyz
            frame_handle.visible = True
            
            label_handle.position = pos
            label_handle.visible = True

def create_robot_control_sliders(
    server: viser.ViserServer,
    viser_urdf: ViserUrdf,
    urdf_model: yourdfpy.URDF,
    handles: dict,
    get_visibility: callable,
    initial_cfg: Optional[np.ndarray] = None,
    use_radians: bool = False,
    on_cfg_update: Optional[callable] = None,
):
    """Creates sliders and attaches callback to update robot AND frames.

    If initial_cfg is provided and has the same length as the number of
    actuated joints, it will be used as the initial joint configuration.

    The sliders display values either in degrees (default) or radians,
    depending on use_radians. Internally, the URDF is always updated in radians.
    """
    slider_handles = []
    joints = viser_urdf.get_actuated_joint_limits()
    
    if not joints:
        return []

    # Wrapper to update both robot and debug visuals
    def update_all(_):
        # Values from sliders are in display units (deg or rad)
        display_cfg = np.array([s.value for s in slider_handles], dtype=float)
        if use_radians:
            cfg = display_cfg
        else:
            cfg = np.deg2rad(display_cfg)

        viser_urdf.update_cfg(cfg)

        # We must also update the underlying yourdfpy model to get new transforms
        urdf_model.update_cfg(cfg)

        # Update markers
        update_frames_and_labels(server, urdf_model, handles, get_visibility())
        if on_cfg_update is not None:
            on_cfg_update(urdf_model)

    num_joints = len(joints)

    for idx, (joint_name, (lower, upper)) in enumerate(joints.items()):
        lower = lower if lower is not None else -np.pi * 2
        upper = upper if upper is not None else np.pi * 2

        # Compute display-space limits
        if use_radians:
            display_lower = lower
            display_upper = upper
            step = 0.01
            unit_label = "rad"
        else:
            display_lower = np.degrees(lower)
            display_upper = np.degrees(upper)
            step = 1.0
            unit_label = "deg"

        # Default initial position is clamped 0.0 in display units
        initial_pos = 0.0
        if display_lower > initial_pos:
            initial_pos = display_lower
        if display_upper < initial_pos:
            initial_pos = display_upper

        # Override from provided initial configuration if valid (also in display units)
        if initial_cfg is not None and len(initial_cfg) == num_joints:
            try:
                initial_pos = float(initial_cfg[idx])
                if initial_pos < display_lower:
                    initial_pos = display_lower
                if initial_pos > display_upper:
                    initial_pos = display_upper
            except (TypeError, ValueError):
                pass

        slider = server.gui.add_slider(
            label=f"{joint_name} ({unit_label})",
            min=display_lower,
            max=display_upper,
            step=step,
            initial_value=initial_pos,
        )
        slider.on_update(update_all)
        slider_handles.append(slider)

    # Apply initial configuration once so the robot matches slider defaults
    if slider_handles:
        update_all(None)

    return slider_handles

def main():
    # Optional: parse initial joint configuration from CLI (6 floats)
    # By default the values are interpreted as degrees; pass --radians to
    # interpret them (and the GUI sliders) in radians instead.
    joint_cfg: Optional[np.ndarray] = None
    use_radians = False

    # CSV-based batch rendering from RobotStudio self-collision results.
    csv_path: Optional[str] = None
    csv_waypoint_configs: Optional[Dict[int, List[Tuple[float, ...]]]] = None
    csv_output_dir: Optional[str] = None

    args = sys.argv[1:]

    # Parse units flag first.
    if "--radians" in args:
        use_radians = True
        args = [a for a in args if a != "--radians"]

    # Parse CSV flag: --csv <path>
    if "--csv" in args:
        idx = args.index("--csv")
        try:
            csv_path = args[idx + 1]
        except IndexError:
            print("Error: --csv flag requires a path to a CSV file.")
            return
        # Remove the flag and its argument from args.
        del args[idx : idx + 2]

    scene_yaml = "config/collision_objects.yaml"
    # Default: every fixture entry in fixture_config.yaml that has an STL.
    # Override with --fixture name[,name2,...] or disable with --no-fixture.
    fixture_names: Optional[List[str]] = None  # None → load all from YAML
    if "--no-scene" in args:
        scene_yaml = None
        args = [a for a in args if a != "--no-scene"]
    if "--scene" in args:
        idx = args.index("--scene")
        try:
            scene_yaml = args[idx + 1]
        except IndexError:
            print("Error: --scene requires a YAML path.")
            return
        del args[idx : idx + 2]
    if "--fixture" in args:
        idx = args.index("--fixture")
        try:
            raw = args[idx + 1]
        except IndexError:
            print(
                "Error: --fixture requires one or more names from "
                "fixture_config.yaml (comma-separated)."
            )
            return
        fixture_names = [n.strip() for n in raw.split(",") if n.strip()]
        if not fixture_names:
            print("Error: --fixture got an empty name list.")
            return
        del args[idx : idx + 2]
    if "--no-fixture" in args:
        fixture_names = []
        args = [a for a in args if a != "--no-fixture"]

    # If CSV is provided, load joint configurations from it.
    if csv_path is not None:
        if not os.path.isabs(csv_path):
            csv_path = os.path.abspath(csv_path)

        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            return

        print(f"Loading self-collision CSV from {csv_path}")

        csv_waypoint_configs = defaultdict(list)

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            required_cols = ["waypoint_index"] + [f"j_{i}" for i in range(1, 7)]
            if reader.fieldnames is None:
                print("Error: CSV has no header.")
                return
            for col in required_cols:
                if col not in reader.fieldnames:
                    print(f"Error: CSV missing required column: {col}")
                    return

            for row in reader:
                try:
                    wp = int(row["waypoint_index"])
                    cfg = tuple(float(row[f"j_{i}"]) for i in range(1, 7))
                except (ValueError, KeyError):
                    continue

                # Store only distinct configurations per waypoint_index.
                if cfg not in csv_waypoint_configs[wp]:
                    csv_waypoint_configs[wp].append(cfg)

        if not csv_waypoint_configs:
            print("Error: No valid waypoint / joint configurations found in CSV.")
            return

        # Use the first waypoint's first configuration as initial pose (in degrees by default).
        first_wp = sorted(csv_waypoint_configs.keys())[0]
        joint_cfg = np.array(csv_waypoint_configs[first_wp][0], dtype=float)
        unit_label = "radians" if use_radians else "degrees"
        print(
            f"Loaded {len(csv_waypoint_configs)} waypoint_indices from CSV. "
            f"Initial joint configuration from waypoint_index {first_wp} "
            f"interpreted as {unit_label}."
        )

        # Derive Experiment_15 folder from CSV path and create self_collision_debug there.
        csv_dir = os.path.dirname(csv_path)
        results_dir = os.path.dirname(csv_dir)
        experiment_dir = os.path.dirname(results_dir)
        csv_output_dir = os.path.abspath(os.path.join(experiment_dir, "self_collision_debug"))
        os.makedirs(csv_output_dir, exist_ok=True)
        print(f"Images from CSV will be saved under: {csv_output_dir}")

    # If no CSV is provided, fall back to single joint configuration from CLI (6 floats).
    if csv_path is None and args:
        raw = " ".join(args)
        tokens = raw.replace(",", " ").split()
        try:
            values = [float(tok) for tok in tokens]
        except ValueError:
            print(
                "Warning: Failed to parse joint states from arguments. "
                "Expected 6 floating-point numbers (space or comma separated)."
            )
            values = []

        if len(values) != 0 and len(values) != 6:
            print(
                f"Warning: Expected 6 joint values, got {len(values)}. "
                "Ignoring provided joint states."
            )
        elif len(values) == 6:
            joint_cfg = np.array(values, dtype=float)
            unit_label = "radians" if use_radians else "degrees"
            print(
                f"Using initial joint configuration from CLI ({unit_label}): "
                f"{joint_cfg}"
            )
    elif csv_path is None:
        unit_label = "radians" if use_radians else "degrees"
        print(f"No joint configuration provided on CLI. Using unit: {unit_label}.")

    # 1. Setup Viser
    server = viser.ViserServer()
    print(f"\nViser server started! Check the banner above for the URL.\n")

    # 2. Load Config
    config = load_config()
    if 'robots' not in config:
        print("Invalid config file structure.")
        return

    robots_map = {r['name']: r for r in config['robots']}
    robot_names = list(robots_map.keys())
    
    if not robot_names:
        print("No robots found in config.")
        return

    # State
    current_urdf_handle = None
    current_sliders = []
    debug_handles = {} # link_name -> [frame_handle, label_handle]
    current_urdf_model: Optional[yourdfpy.URDF] = None
    joint_state_label = None
    # Each entry: (mesh_handle, T_parent_fixture, parent_link)
    fixture_parts: List[Tuple[Any, np.ndarray, str]] = []
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if fixture_names is None:
        fixture_names = list_fixture_names(repo_root)
    print(
        f"Fixture meshes to load ({len(fixture_names)}): "
        f"{', '.join(fixture_names) if fixture_names else '(none)'}"
    )

    def update_fixture_pose(urdf_model: yourdfpy.URDF) -> None:
        if not fixture_parts:
            return
        for handle, T_link_fixture, parent_link in fixture_parts:
            try:
                T_world_link = urdf_model.scene.graph.get(parent_link)[0]
            except Exception:
                continue
            T_world = np.asarray(T_world_link, dtype=float) @ T_link_fixture
            pos, wxyz = _T_to_pos_wxyz(T_world)
            handle.position = pos
            handle.wxyz = wxyz

    # 3. Callbacks
    def load_robot(name):
        nonlocal current_urdf_handle, current_sliders, debug_handles, joint_cfg, current_urdf_model
        nonlocal fixture_parts
        
        # Clear old handles
        for s in current_sliders: s.remove()
        current_sliders = []
        
        # Clear scene (resets frames/labels too)
        server.scene.reset()
        debug_handles = {}
        fixture_parts = []
        
        server.scene.add_grid("floor", width=10, height=10)
        server.scene.add_frame("world_frame", show_axes=True, axes_length=0.2)

        # Load New
        robot_conf = robots_map[name]
        urdf_path = robot_conf['urdf_path']
        
        print(f"Loading {name} from {urdf_path}...")
        if not os.path.exists(urdf_path):
             urdf_path = os.path.abspath(urdf_path)
        
        if not os.path.exists(urdf_path):
            print(f"Error: URDF not found: {urdf_path}")
            return

        try:
            # Load URDF
            urdf_model = yourdfpy.URDF.load(
                urdf_path,
                build_scene_graph=True,
                load_meshes=True,
                load_collision_meshes=False
            )

            # ViserUrdf
            current_urdf_handle = ViserUrdf(
                server,
                urdf_or_path=urdf_model,
                load_meshes=True,
                load_collision_meshes=False
            )
            current_urdf_model = urdf_model

            if scene_yaml:
                scene_path = scene_yaml if os.path.isabs(scene_yaml) else os.path.join(repo_root, scene_yaml)
                add_collision_scene_meshes(server, scene_path, repo_root)

            if fixture_names:
                fixture_parts = add_fixture_meshes(server, fixture_names, repo_root)
                update_fixture_pose(urdf_model)
            
            # Initial Frames Update
            update_frames_and_labels(server, urdf_model, debug_handles, show_frames_cb.value)

            # Sliders
            with server.gui.add_folder("Joint Control"):
                current_sliders = create_robot_control_sliders(
                    server,
                    current_urdf_handle,
                    urdf_model,
                    debug_handles,
                    lambda: show_frames_cb.value,
                    initial_cfg=joint_cfg,
                    use_radians=use_radians,
                    on_cfg_update=update_fixture_pose,
                )
                
        except Exception as e:
            print(f"Error loading robot: {e}")
            import traceback
            traceback.print_exc()

    # 4. GUI Controls
    with server.gui.add_folder("Settings"):
        dropdown = server.gui.add_dropdown(
            label="Robot Model",
            options=robot_names,
            initial_value=robot_names[0]
        )
        
        show_frames_cb = server.gui.add_checkbox(
            "Show Frames & Labels",
            initial_value=True
        )
        
        reset_btn = server.gui.add_button("Reset Scene")

        # Text area to show current joint configuration (top-right GUI area).
        joint_state_label = server.gui.add_markdown("Joint state: N/A")

    # Toggle Callback
    def on_toggle_frames(_):
        if current_urdf_handle and current_urdf_handle._urdf: # access internal yourdfpy model if stored?
            # actually we need the urdf_model object. 
            # ViserUrdf stores it as _urdf usually.
            model = current_urdf_handle._urdf
            update_frames_and_labels(server, model, debug_handles, show_frames_cb.value)
            
    show_frames_cb.on_update(on_toggle_frames)
    dropdown.on_update(lambda _: load_robot(dropdown.value))
    reset_btn.on_click(lambda _: load_robot(dropdown.value))

    # If a CSV of self-collision results is provided, add manual controls: choose a
    # config (waypoint + variant), position the camera, then click "Save current image".
    if csv_waypoint_configs is not None and csv_output_dir is not None:
        # Flat list: (waypoint_index, config_index, cfg_tuple) for dropdown and saving
        csv_flat_list: List[Tuple[int, int, Tuple[float, ...]]] = []
        for wp in sorted(csv_waypoint_configs.keys()):
            cfgs = csv_waypoint_configs[wp]
            for idx, cfg in enumerate(cfgs):
                csv_flat_list.append((wp, idx, cfg))
        csv_dropdown_options = [
            f"Waypoint {wp}" + (f" config {idx}" if len(csv_waypoint_configs[wp]) > 1 else "")
            for wp, idx, _ in csv_flat_list
        ]
        csv_current_index = 0

        def _apply_csv_config(index: int) -> None:
            nonlocal current_urdf_handle, current_urdf_model, joint_state_label
            if not csv_flat_list or current_urdf_handle is None or current_urdf_model is None:
                return
            index = max(0, min(index, len(csv_flat_list) - 1))
            waypoint_index, _, cfg_display = csv_flat_list[index]
            cfg_display_arr = np.array(cfg_display, dtype=float)
            cfg_rad = cfg_display_arr if use_radians else np.deg2rad(cfg_display_arr)
            current_urdf_handle.update_cfg(cfg_rad)
            current_urdf_model.update_cfg(cfg_rad)
            update_frames_and_labels(
                server, current_urdf_model, debug_handles, show_frames_cb.value
            )
            update_fixture_pose(current_urdf_model)
            unit = "rad" if use_radians else "deg"
            if joint_state_label is not None:
                text = f"Waypoint {waypoint_index} | units: {unit}\n"
                text += ", ".join(
                    [f"j_{i+1}={cfg_display_arr[i]:.3f} {unit}" for i in range(6)]
                )
                joint_state_label.content = text

        with server.gui.add_folder("CSV config (save when ready)"):
            csv_dropdown = server.gui.add_dropdown(
                label="Current config",
                options=csv_dropdown_options,
                initial_value=csv_dropdown_options[0],
            )

            def _on_csv_dropdown_change(_) -> None:
                nonlocal csv_current_index
                val = csv_dropdown.value
                try:
                    i = csv_dropdown_options.index(val)
                    csv_current_index = i
                    _apply_csv_config(i)
                except (ValueError, IndexError):
                    pass

            csv_dropdown.on_update(_on_csv_dropdown_change)

            save_current_btn = server.gui.add_button("Save current image")

            def _on_save_current_click(event) -> None:
                nonlocal csv_current_index
                client = getattr(event, "client", None)
                if client is None:
                    clients = server.get_clients()
                    client = next(iter(clients.values()), None) if clients else None
                if client is None:
                    print("No client. Open the viser URL in a browser first.")
                    return
                if not csv_flat_list or current_urdf_handle is None:
                    print("No CSV config or robot loaded.")
                    return
                waypoint_index, idx, _ = csv_flat_list[csv_current_index]
                cfgs = csv_waypoint_configs[waypoint_index]
                multiple = len(cfgs) > 1
                filename = f"{waypoint_index}_{idx}.png" if multiple else f"{waypoint_index}.png"
                out_path = os.path.join(csv_output_dir, filename)
                try:
                    img = client.get_render(height=720, width=1280)
                    iio.imwrite(out_path, img)
                    print(f"Saved: {out_path}")
                except Exception as e:
                    print(f"Failed to save: {e}")
                    import traceback
                    traceback.print_exc()

            save_current_btn.on_click(_on_save_current_click)

        # Apply first CSV config so the robot matches the dropdown on load
        _apply_csv_config(0)

    # Initial Load
    load_robot(robot_names[0])

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
