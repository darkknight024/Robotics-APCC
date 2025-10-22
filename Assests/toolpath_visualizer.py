# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Toolpath Visualizer for Isaac Lab

This script visualizes cutting toolpaths by:
- Creating a cube representing the workpiece to be cut (30cm x 10cm x 5cm)
- Loading toolpath data from CSV (in millimeters, local coordinates)
- Displaying waypoints as small spheres on the cube surface

Usage:
    python toolpath_visualizer.py --csv path/to/toolpath.csv
"""

import argparse
import os
import pandas as pd
import numpy as np

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Toolpath Visualizer for Cutting Simulation")
parser.add_argument("--csv", type=str, required=True,
                    help="Path to CSV file containing toolpath data (x,y,z in millimeters)")
parser.add_argument("--mode", type=str, default="v", choices=["v", "e"], 
                    help="Mode: 'v' for view (visualization), 'e' for export (convert coordinates)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Validate arguments
if not os.path.exists(args_cli.csv):
    parser.error(f"CSV file not found: {args_cli.csv}")

# For export mode, we don't need Isaac Lab
if args_cli.mode == "e":
    print("[INFO] Export mode - running coordinate conversion without Isaac Lab")
    # Import only what we need for export
    import pandas as pd
    import numpy as np
    
    # Define coordinate conversion function for export mode
    def convert_local_to_global_export(csv_x_mm, csv_y_mm, csv_z_mm, cube_center):
        """Convert local CSV coordinates (in mm) to global Isaac Lab coordinates (in m) for export."""
        # Validate coordinate ranges and warn if out of bounds
        if not (0 <= csv_x_mm <= 100):
            print(f"[WARNING] Local X coordinate {csv_x_mm}mm is outside expected range [0, 100]mm")
        if not (0 <= csv_y_mm <= 300):
            print(f"[WARNING] Local Y coordinate {csv_y_mm}mm is outside expected range [0, 300]mm")
        if not (0 <= csv_z_mm <= 50):
            print(f"[WARNING] Local Z coordinate {csv_z_mm}mm is outside expected range [0, 50]mm")
        
        # Convert mm to meters
        csv_x_m = csv_x_mm / 1000.0
        csv_y_m = csv_y_mm / 1000.0
        csv_z_m = csv_z_mm / 1000.0
        
        # Calculate origin positions for each axis in global coordinates
        # Local origin is at bottom-left corner of cube
        origin_global_x = cube_center[0] - 0.3 / 2.0   # -0.15m (length/2)
        origin_global_y = cube_center[1] - 0.1 / 2.0   # -0.05m (width/2)
        origin_global_z = cube_center[2] - 0.05 / 2.0  # -0.025m (height/2)
        
        # Map local coordinates to global coordinates:
        # Local X (width, 0-100mm) -> Global Y (width dimension)
        # Local Y (length, 0-300mm) -> Global X (length dimension)
        # Local Z (height, 0-50mm) -> Global Z (height dimension)
        global_x = origin_global_x + csv_y_m  # Local Y becomes Global X
        global_y = origin_global_y + csv_x_m  # Local X becomes Global Y
        global_z = origin_global_z + csv_z_m  # Local Z becomes Global Z
        
        return global_x, global_y, global_z
    
    def convert_quaternion_to_end_effector_frame(qw, qx, qy, qz):
        """
        Apply CSV rotation RELATIVE TO the fixed base rotation (pointing down).
        
        Base rotation (end-effector pointing down):
        - Roll (X): 0° = 0 rad
        - Pitch (Y): 90° = 1.5708 rad  
        - Yaw (Z): 0° = 0 rad
        
        Then applies CSV rotation on top of this base rotation.
        Final rotation = base_rotation * csv_rotation
        """
        import torch
        import math
        import isaaclab.utils.math as math_utils
        
        # Fixed base rotation: Roll=0°, Pitch=90°, Yaw=0° (pointing down)
        base_roll = torch.tensor([0.0])           # 0 degrees
        base_pitch = torch.tensor([math.pi / 2])  # 90 degrees (pointing down)
        base_yaw = torch.tensor([0.0])            # 0 degrees
        
        # Convert base rotation to quaternion
        base_quat = math_utils.quat_from_euler_xyz(base_roll, base_pitch, base_yaw)  # Returns (w, x, y, z)
        base_qw, base_qx, base_qy, base_qz = base_quat[0, 0], base_quat[0, 1], base_quat[0, 2], base_quat[0, 3]
        
        # CSV rotation as quaternion (already provided)
        csv_qw, csv_qx, csv_qy, csv_qz = qw, qx, qy, qz
        
        # Quaternion multiplication: q_final = q_base * q_csv
        # This applies CSV rotation relative to the base rotation
        final_qw = base_qw * csv_qw - base_qx * csv_qx - base_qy * csv_qy - base_qz * csv_qz
        final_qx = base_qw * csv_qx + base_qx * csv_qw + base_qy * csv_qz - base_qz * csv_qy
        final_qy = base_qw * csv_qy - base_qx * csv_qz + base_qy * csv_qw + base_qz * csv_qx
        final_qz = base_qw * csv_qz + base_qx * csv_qy - base_qy * csv_qx + base_qz * csv_qw
        
        # Convert back to float values
        result_qw = final_qw.item() if hasattr(final_qw, 'item') else float(final_qw)
        result_qx = final_qx.item() if hasattr(final_qx, 'item') else float(final_qx)
        result_qy = final_qy.item() if hasattr(final_qy, 'item') else float(final_qy)
        result_qz = final_qz.item() if hasattr(final_qz, 'item') else float(final_qz)
        
        print(f"[DEBUG] Base quaternion (Pointing down): ({base_qw:.6f}, {base_qx:.6f}, {base_qy:.6f}, {base_qz:.6f})")
        print(f"[DEBUG] CSV rotation: ({csv_qw:.6f}, {csv_qx:.6f}, {csv_qy:.6f}, {csv_qz:.6f})")
        print(f"[DEBUG] Final rotation (Base * CSV): ({result_qw:.6f}, {result_qx:.6f}, {result_qy:.6f}, {result_qz:.6f})")
        
        return result_qw, result_qx, result_qy, result_qz

    def export_converted_coordinates():
        """Export mode: Convert CSV coordinates to global world positions."""
        print(f"\n[INFO] Export Mode - Converting coordinates from: {args_cli.csv}")
        
        # Get cube origin from user
        print(f"\n[INFO] Please specify the cube origin position in meters (global world coordinates):")
        try:
            cube_x = float(input("Cube center X (meters): "))
            cube_y = float(input("Cube center Y (meters): "))  
            cube_z = float(input("Cube center Z (meters): "))
            cube_origin = (cube_x, cube_y, cube_z)
            print(f"[INFO] Using cube origin: ({cube_x}, {cube_y}, {cube_z}) meters")
        except ValueError:
            print("[ERROR] Invalid input. Please enter numeric values.")
            return
        
        # Load CSV data
        try:
            df = pd.read_csv(args_cli.csv)
            print(f"[INFO] CSV columns: {df.columns.tolist()}")
            print(f"[INFO] Total rows in CSV: {len(df)}")
            
            # Extract coordinates and quaternions, create 7-column output
            converted_data = []
            
            for idx, row in df.iterrows():
                try:
                    # Extract position (first 3 columns)
                    local_x = float(row.iloc[0])
                    local_y = float(row.iloc[1]) 
                    local_z = float(row.iloc[2])
                    
                    # Extract quaternion (assuming columns 3-6 are qw, qx, qy, qz)
                    if len(row) >= 7:
                        orig_qw = float(row.iloc[3])
                        orig_qx = float(row.iloc[4])
                        orig_qy = float(row.iloc[5])
                        orig_qz = float(row.iloc[6])
                    else:
                        print(f"[WARNING] Row {idx} missing quaternion data, using identity quaternion")
                        orig_qw, orig_qx, orig_qy, orig_qz = 1.0, 0.0, 0.0, 0.0
                    
                    # Convert position to global coordinates
                    global_x, global_y, global_z = convert_local_to_global_export(
                        local_x, local_y, local_z, cube_origin
                    )
                    
                    # Convert quaternion to end-effector frame (Roll=-180°, Pitch=+90°, Yaw=0°)
                    new_qw, new_qx, new_qy, new_qz = convert_quaternion_to_end_effector_frame(
                        orig_qw, orig_qx, orig_qy, orig_qz
                    )
                    
                    # Create 7-column row: [x, y, z, qw, qx, qy, qz]
                    converted_row = [global_x, global_y, global_z, new_qw, new_qx, new_qy, new_qz]
                    converted_data.append(converted_row)
                    
                except (ValueError, TypeError, IndexError):
                    print(f"[INFO] Skipping row {idx}: {row.iloc[0]} (invalid data)")
                    continue
            
            if len(converted_data) == 0:
                print("[ERROR] No valid waypoints found in CSV!")
                return
            
            print(f"[INFO] Converted {len(converted_data)} waypoints")
            
            # Create output dataframe with 7 columns
            column_names = ["x", "y", "z", "qw", "qx", "qy", "qz"]
            final_df = pd.DataFrame(converted_data, columns=column_names)
            
            # Generate output filename
            input_dir = os.path.dirname(args_cli.csv)
            input_name = os.path.splitext(os.path.basename(args_cli.csv))[0]
            output_filename = f"{input_name}_global_coords.csv"
            output_path = os.path.join(input_dir, output_filename)
            
            # Save converted CSV
            final_df.to_csv(output_path, index=False)
            
            print(f"\n[INFO] ✓ Export complete!")
            print(f"[INFO] Input file: {args_cli.csv}")
            print(f"[INFO] Output file: {output_path}")
            print(f"[INFO] Cube origin used: ({cube_x}, {cube_y}, {cube_z}) meters")
            print(f"[INFO] Converted {len(converted_data)} waypoints from local (mm) to global (m)")
            print(f"[INFO] Applied base rotation (Roll=0°, Pitch=90°, Yaw=0°) + CSV relative rotations")
            
            # Show sample of conversion
            if len(converted_data) > 0:
                first_local = [float(df.iloc[0, 0]), float(df.iloc[0, 1]), float(df.iloc[0, 2])]
                first_global = converted_data[0]
                orig_quat = [float(df.iloc[0, 3]), float(df.iloc[0, 4]), float(df.iloc[0, 5]), float(df.iloc[0, 6])]
                new_quat = first_global[3:7]
                
                print(f"\n[INFO] Example conversion:")
                print(f"       Local pos (mm): ({first_local[0]:.2f}, {first_local[1]:.2f}, {first_local[2]:.2f})")
                print(f"       Global pos (m): ({first_global[0]:.6f}, {first_global[1]:.6f}, {first_global[2]:.6f})")
                print(f"       Original quat (CSV): ({orig_quat[0]:.6f}, {orig_quat[1]:.6f}, {orig_quat[2]:.6f}, {orig_quat[3]:.6f})")
                print(f"       Final quat (Base + CSV): ({new_quat[0]:.6f}, {new_quat[1]:.6f}, {new_quat[2]:.6f}, {new_quat[3]:.6f})")
            
        except Exception as e:
            print(f"[ERROR] Failed to process CSV: {e}")
            return
    
    # Run export and exit
    export_converted_coordinates()
    exit(0)

# Launch omniverse app (only for visualization mode)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Workpiece cube dimensions (in meters)
# This represents the material to be cut
CUBE_LENGTH = 0.3   # 30 cm (X-axis in global coordinates)
CUBE_WIDTH = 0.1    # 10 cm (Y-axis in global coordinates)  
CUBE_HEIGHT = 0.05  # 5 cm (Z-axis in global coordinates)

# Cube position in world (centered at origin for simplicity)
CUBE_POSITION = (0.0, 0.0, 0.0)

# Sphere marker configuration
SPHERE_RADIUS = 0.005  # 5mm radius (visible but small)
SPHERE_COLOR = (0.0, 1.0, 1.0)  # Cyan color for visibility

# Origin marker configuration
ORIGIN_SPHERE_RADIUS = 0.01  # 10mm radius (larger for visibility)
ORIGIN_SPHERE_COLOR = (1.0, 1.0, 0.0)  # Yellow color

# ============================================================================
# SCENE CONFIGURATION
# ============================================================================

def create_dynamic_scene_with_spheres(waypoints_global, origin_pos):
    """Create scene configuration with dynamically added spheres."""
    
    @configclass
    class DynamicToolpathSceneCfg(InteractiveSceneCfg):
        """Configuration for toolpath visualization scene with dynamic spheres."""
        
        # Ground plane
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )
        
        # Lights
        dome_light = AssetBaseCfg(
            prim_path="/World/Light", 
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )
        
        # Workpiece cube
        workpiece = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Workpiece",
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_LENGTH, CUBE_WIDTH, CUBE_HEIGHT),
                collision_props=None,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.8, 0.8),  # Light gray
                    # metallic=0.3,
                    # roughness=0.7,
                    # opacity removed as requested
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=CUBE_POSITION),
        )
        
        # Yellow origin sphere
        origin_sphere = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/OriginSphere",
            spawn=sim_utils.SphereCfg(
                radius=ORIGIN_SPHERE_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=ORIGIN_SPHERE_COLOR,
                    metallic=0.0,
                    roughness=0.3
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=origin_pos),
        )
    
    # Dynamically add waypoint spheres
    for i, waypoint in enumerate(waypoints_global):
        sphere_attr = f"waypoint_sphere_{i}"
        setattr(DynamicToolpathSceneCfg, sphere_attr, AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/WaypointSphere_{i}",
            spawn=sim_utils.SphereCfg(
                radius=SPHERE_RADIUS,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=SPHERE_COLOR,
                    metallic=0.0,
                    roughness=0.5
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(waypoint[0], waypoint[1], waypoint[2])),
        ))
    
    return DynamicToolpathSceneCfg

# ============================================================================
# COORDINATE CONVERSION
# ============================================================================

def convert_local_to_global(csv_x_mm, csv_y_mm, csv_z_mm, cube_center):
    """
    Convert local CSV coordinates (in mm) to global Isaac Lab coordinates (in m).
    
    CSV Local Coordinates:
    - Origin (0, 0, 0) is at the BOTTOM-LEFT corner of the cube
    - X: 0 to 100mm (width dimension - global Y axis)
    - Y: 0 to 300mm (length dimension - global X axis)  
    - Z: 0 to 50mm (height dimension - global Z axis)
    
    Global Coordinates:
    - Cube centered at cube_center
    - X: length dimension (0.3m = 300mm)
    - Y: width dimension (0.1m = 100mm)
    - Z: height dimension (0.05m = 50mm)
    
    Args:
        csv_x_mm: Local X coordinate in millimeters (0-100, maps to global Y)
        csv_y_mm: Local Y coordinate in millimeters (0-300, maps to global X)
        csv_z_mm: Local Z coordinate in millimeters (0-50, maps to global Z)
        cube_center: Tuple (x, y, z) of cube center in global coordinates
        
    Returns:
        Tuple (global_x, global_y, global_z) in meters
    """
    # Validate coordinate ranges and warn if out of bounds
    if not (0 <= csv_x_mm <= 100):
        print(f"[WARNING] Local X coordinate {csv_x_mm}mm is outside expected range [0, 100]mm")
    if not (0 <= csv_y_mm <= 300):
        print(f"[WARNING] Local Y coordinate {csv_y_mm}mm is outside expected range [0, 300]mm")
    if not (0 <= csv_z_mm <= 50):
        print(f"[WARNING] Local Z coordinate {csv_z_mm}mm is outside expected range [0, 50]mm")
    
    # Convert mm to meters
    csv_x_m = csv_x_mm / 1000.0
    csv_y_m = csv_y_mm / 1000.0
    csv_z_m = csv_z_mm / 1000.0
    
    # Calculate origin positions for each axis in global coordinates
    # Local origin is at bottom-left corner of cube
    origin_global_x = cube_center[0] - CUBE_LENGTH / 2.0   # -0.15m (length/2)
    origin_global_y = cube_center[1] - CUBE_WIDTH / 2.0    # -0.05m (width/2)
    origin_global_z = cube_center[2] - CUBE_HEIGHT / 2.0   # -0.025m (height/2)
    
    # Map local coordinates to global coordinates:
    # Local X (width, 0-100mm) -> Global Y (width dimension)
    # Local Y (length, 0-300mm) -> Global X (length dimension)
    # Local Z (height, 0-50mm) -> Global Z (height dimension)
    global_x = origin_global_x + csv_y_m  # Local Y becomes Global X
    global_y = origin_global_y + csv_x_m  # Local X becomes Global Y
    global_z = origin_global_z + csv_z_m  # Local Z becomes Global Z
    
    return global_x, global_y, global_z

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function."""
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set camera view (position the camera to see the cube)
    sim.set_camera_view([0.5, 0.5, 0.3], [0.0, 0.0, 0.0])
    
    # First, load and process the CSV data
    print(f"\n[INFO] Loading toolpath from: {args_cli.csv}")
    
    try:
        df = pd.read_csv(args_cli.csv)
        print(f"[INFO] CSV columns: {df.columns.tolist()}")
        print(f"[INFO] Total rows in CSV: {len(df)}")
        
        # Debug: Show first few raw rows with column names
        print(f"\n[DEBUG] First 5 raw CSV rows:")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            print(f"[DEBUG] Raw row {i}: x={row['x']}, y={row['y']}, z={row['z']}")
        
        # Extract X, Y, Z coordinates using column names
        # Filter out any non-numeric rows (like "T0" markers)
        valid_rows = []
        for idx, row in df.iterrows():
            try:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                valid_rows.append((x, y, z))
                # Debug: Show first few rows being loaded
                if len(valid_rows) <= 5:
                    print(f"[DEBUG] Row {idx}: Loading ({x:.2f}, {y:.2f}, {z:.2f})")
            except (ValueError, TypeError):
                print(f"[INFO] Skipping row {idx}: {row['x']} (non-numeric)")
                continue
        
        if len(valid_rows) == 0:
            print("[ERROR] No valid numeric waypoints found in CSV!")
            return
            
        waypoints_local = np.array(valid_rows)
        print(f"[INFO] Loaded {len(waypoints_local)} valid waypoints")
        print(f"[INFO] Local coordinate ranges (mm):")
        print(f"       X: [{waypoints_local[:, 0].min():.2f}, {waypoints_local[:, 0].max():.2f}]")
        print(f"       Y: [{waypoints_local[:, 1].min():.2f}, {waypoints_local[:, 1].max():.2f}]")
        print(f"       Z: [{waypoints_local[:, 2].min():.2f}, {waypoints_local[:, 2].max():.2f}]")
        
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return
    
    # Convert local coordinates to global coordinates
    print(f"\n[INFO] Converting coordinates from local (mm) to global (m)...")
    waypoints_global = []
    for x_mm, y_mm, z_mm in waypoints_local:
        global_x, global_y, global_z = convert_local_to_global(
            x_mm, y_mm, z_mm, CUBE_POSITION
        )
        waypoints_global.append([global_x, global_y, global_z])
    
    waypoints_global = np.array(waypoints_global)
    print(f"[INFO] Global coordinate ranges (m):")
    print(f"       X: [{waypoints_global[:, 0].min():.3f}, {waypoints_global[:, 0].max():.3f}]")
    print(f"       Y: [{waypoints_global[:, 1].min():.3f}, {waypoints_global[:, 1].max():.3f}]")
    print(f"       Z: [{waypoints_global[:, 2].min():.3f}, {waypoints_global[:, 2].max():.3f}]")
    
    # Debug: Show first waypoint conversion step by step
    if len(waypoints_global) > 0:
        first_local = waypoints_local[0]
        first_global = waypoints_global[0]
        
        print(f"\n[DEBUG] Step-by-step conversion for first waypoint:")
        print(f"        CSV Row: {first_local[0]:.2f}, {first_local[1]:.2f}, {first_local[2]:.2f} (first 3 numbers only)")
        print(f"        ")
        print(f"        Step 1 - Local coordinates in mm:")
        print(f"        Local X (width): {first_local[0]:.2f}mm")
        print(f"        Local Y (length): {first_local[1]:.2f}mm") 
        print(f"        Local Z (height): {first_local[2]:.2f}mm")
        print(f"        ")
        print(f"        Step 2 - Calculate origins in mm:")
        print(f"        Origin X: -300/2 = -150mm")
        print(f"        Origin Y: -100/2 = -50mm")
        print(f"        Origin Z: -50/2 = -25mm")
        print(f"        ")
        print(f"        Step 3 - Add local to origins (still in mm):")
        print(f"        Result X = Origin X + Local Y = -150 + {first_local[1]:.2f} = {-150 + first_local[1]:.2f}mm")
        print(f"        Result Y = Origin Y + Local X = -50 + {first_local[0]:.2f} = {-50 + first_local[0]:.2f}mm")
        print(f"        Result Z = Origin Z + Local Z = -25 + {first_local[2]:.2f} = {-25 + first_local[2]:.2f}mm")
        print(f"        ")
        print(f"        Step 4 - Axis swap and convert to meters:")
        expected_global_x = (-150 + first_local[1]) / 1000.0  # Local Y -> Global X
        expected_global_y = (-50 + first_local[0]) / 1000.0   # Local X -> Global Y  
        expected_global_z = (-25 + first_local[2]) / 1000.0   # Local Z -> Global Z
        print(f"        Global X = {(-150 + first_local[1]):.2f}mm = {expected_global_x:.3f}m")
        print(f"        Global Y = {(-50 + first_local[0]):.2f}mm = {expected_global_y:.3f}m")
        print(f"        Global Z = {(-25 + first_local[2]):.2f}mm = {expected_global_z:.3f}m")
        print(f"        ")
        print(f"        Final Expected: ({expected_global_x:.3f}, {expected_global_y:.3f}, {expected_global_z:.3f}) meters")
        print(f"        Actual Result:  ({first_global[0]:.3f}, {first_global[1]:.3f}, {first_global[2]:.3f}) meters")
        
        # Verify calculation matches
        if (abs(first_global[0] - expected_global_x) < 0.001 and 
            abs(first_global[1] - expected_global_y) < 0.001 and
            abs(first_global[2] - expected_global_z) < 0.001):
            print(f"        ✅ Coordinate conversion is CORRECT")
        else:
            print(f"        ❌ Coordinate conversion MISMATCH!")
    
    # Convert local origin (0,0,0) to global coordinates for yellow sphere
    origin_global_x, origin_global_y, origin_global_z = convert_local_to_global(
        0.0, 0.0, 0.0, CUBE_POSITION
    )
    print(f"\n[INFO] Local origin (0,0,0) -> Global ({origin_global_x:.3f}, {origin_global_y:.3f}, {origin_global_z:.3f})")
    
    # Create dynamic scene with all spheres
    print(f"\n[INFO] Creating scene with {len(waypoints_global)} spheres...")
    scene_cfg_class = create_dynamic_scene_with_spheres(
        waypoints_global, 
        (origin_global_x, origin_global_y, origin_global_z)
    )
    scene_cfg = scene_cfg_class(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Reset simulation
    sim.reset()
    print("[INFO] Scene setup complete...")
    
    print(f"\n[INFO] ✓ Visualization complete!")
    print(f"[INFO] Cube dimensions: {CUBE_LENGTH}m x {CUBE_WIDTH}m x {CUBE_HEIGHT}m")
    print(f"[INFO] Cube position: {CUBE_POSITION}")
    print(f"[INFO] Yellow origin sphere: Local (0,0,0) at Global ({origin_global_x:.3f}, {origin_global_y:.3f}, {origin_global_z:.3f})")
    print(f"[INFO] Cyan toolpath spheres: {len(waypoints_global)} waypoints (radius: {SPHERE_RADIUS}m)")
    print(f"\n[INFO] Scene is now running. Close the window to exit.")
    
    # Run simulation (just idle, no physics needed)
    sim_dt = sim.get_physics_dt()
    count = 0
    
    while simulation_app.is_running():
        # Step simulation
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        # Print status occasionally
        if count % 1000 == 0:
            print(f"[INFO] Simulation running... (step {count})", end="\r")


if __name__ == "__main__":
    # Run main function
    main()
    # Close simulation app
    simulation_app.close()

