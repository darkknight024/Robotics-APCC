#!/usr/bin/env python3
"""
Interactive 3D Zone Segment Visualizer
========================================

Generates an interactive 3D Plotly visualization of the programmed path
with blend arcs, showing:

  * Straight segments (blue lines)
  * Blend arcs (orange arcs per zone)
  * Input waypoint poses (green markers with orientation axes)
  * Zone annotations: r_eff, path deviation max, arc-length parameterized
    coordinates [x,y,z] and quaternion [qw,qx,qy,qz] along the dense path

Supports two input modes:
  * Toolpath in T_P_K frame → transformed via knife pose (default: Zund)
  * Waypoints in robot base frame → use ``--base_frame``

Zone data must be preset zone numbers (z0, z1, z5, z10, …).

Usage::

    cd iue/

    # Siping toolpath (Zund knife transform, default)
    conda run -n robotics python -m core.blend_zone.plot_zone_segments \\
        --csv path/to/siping_toolpath.csv --output output_dir/

    # Corner waypoints (base frame)
    conda run -n robotics python -m core.blend_zone.plot_zone_segments \\
        --csv path/to/corner_60_deg_v500_z5.csv --base_frame --output output_dir/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.csv_loader_toolpath import load_toolpath_f3
from core.blend_zone.zone_resolver import (
    resolve_zone_list,
    apply_overlap_reduction,
    ZoneParams,
)
from core.blend_zone.blend_geometry import compute_blend_geometries
from core.blend_zone.path_sampler import sample_blended_path


def _quaternion_to_axes(q: np.ndarray, scale: float = 2.0):
    """Convert quaternion [qw,qx,qy,qz] to 3 axis vectors for visualization."""
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])
    return R[:, 0] * scale, R[:, 1] * scale, R[:, 2] * scale


def generate_zone_visualization(
    csv_path: str,
    output_dir: str,
    base_frame: bool = False,
    knife_name: str = "Zund",
    traj_index: int = 0,
    ds_mm: float = 0.5,
):
    """Generate interactive 3D zone segment visualization.

    Args:
        csv_path:    Path to toolpath CSV with zone data.
        output_dir:  Directory for output HTML and PNGs.
        base_frame:  True if waypoints are already in robot base frame.
        knife_name:  Knife pose name for frame transformation.
        traj_index:  Which trajectory to visualize (0-based).
        ds_mm:       Arc-length sampling resolution for dense path.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("ERROR: plotly required. Install with: pip install plotly")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load toolpath
    result = load_toolpath_f3(csv_path, custom_zone=False)
    if traj_index >= len(result.waypoints):
        print(f"Trajectory index {traj_index} out of range (have {len(result.waypoints)})")
        return

    waypoints = result.waypoints[traj_index]
    zone_specs = result.zone_specs[traj_index]
    v_cmd = result.v_cmd[traj_index]

    # Apply knife transform if needed
    if not base_frame:
        from utils.config_loader import load_knife_config
        knives = load_knife_config("config/knife_config.yaml")
        if knife_name not in knives:
            print(f"Knife pose '{knife_name}' not found")
            return
        knife = knives[knife_name]
        from utils.transform_handler import transform_trajectory_to_base_frame
        waypoints = transform_trajectory_to_base_frame(
            waypoints, knife.translation_m, knife.quaternion
        )

    n_wp = len(waypoints)
    print(f"Loaded trajectory {traj_index}: {n_wp} waypoints")

    # Resolve zones
    zones = resolve_zone_list(zone_specs)
    zones = apply_overlap_reduction(zones, waypoints)

    # Compute blend geometries
    blend_geoms = compute_blend_geometries(waypoints, zones)

    # Build dense path
    dense_path = sample_blended_path(waypoints, zones, blend_geoms, v_cmd, ds_mm=ds_mm)

    # Positions in mm for plotting
    wp_mm = waypoints[:, :3] * 1000.0
    dense_mm = dense_path.poses[:, :3] * 1000.0

    # ── Build Plotly figure ──
    fig = go.Figure()

    # Dense path colored by segment type
    straight_mask = ~dense_path.is_blend_arc
    blend_mask = dense_path.is_blend_arc

    if np.any(straight_mask):
        fig.add_trace(go.Scatter3d(
            x=dense_mm[straight_mask, 0], y=dense_mm[straight_mask, 1],
            z=dense_mm[straight_mask, 2],
            mode="lines", line=dict(color="royalblue", width=3),
            name="Straight segments",
        ))

    if np.any(blend_mask):
        fig.add_trace(go.Scatter3d(
            x=dense_mm[blend_mask, 0], y=dense_mm[blend_mask, 1],
            z=dense_mm[blend_mask, 2],
            mode="markers", marker=dict(color="orange", size=2),
            name="Blend arcs",
        ))

    # Input waypoints with hover info
    hover_texts = []
    zone_summary = []
    for i in range(n_wp):
        zp = zones[i]
        hover = (
            f"WP {i}<br>"
            f"pos: [{wp_mm[i,0]:.1f}, {wp_mm[i,1]:.1f}, {wp_mm[i,2]:.1f}] mm<br>"
            f"quat: [{waypoints[i,3]:.4f}, {waypoints[i,4]:.4f}, "
            f"{waypoints[i,5]:.4f}, {waypoints[i,6]:.4f}]<br>"
            f"zone: {zp.source} (fine={zp.finep})<br>"
            f"pzone_tcp: {zp.pzone_tcp_mm:.1f} mm<br>"
            f"r_eff: {zp.eff_pzone_tcp_mm:.2f} mm<br>"
            f"v_cmd: {v_cmd[i]:.0f} mm/s"
        )
        hover_texts.append(hover)

        # Path deviation (corner cutting) for interior waypoints with blend.
        # The blend arc for waypoint i sits between segments (i-1) and i.
        # Deviation = distance from programmed waypoint to nearest blend arc point.
        dev_max = 0.0
        if not zp.finep and i > 0 and i < n_wp - 1:
            arc_mask = dense_path.is_blend_arc & (
                (dense_path.segment_ids == i - 1) | (dense_path.segment_ids == i)
            )
            blend_samples = dense_mm[arc_mask]
            if len(blend_samples) > 0:
                dists = np.linalg.norm(blend_samples - wp_mm[i], axis=1)
                dev_max = float(np.min(dists))

        zone_summary.append({
            "waypoint": i,
            "x_mm": float(wp_mm[i, 0]),
            "y_mm": float(wp_mm[i, 1]),
            "z_mm": float(wp_mm[i, 2]),
            "qw": float(waypoints[i, 3]),
            "qx": float(waypoints[i, 4]),
            "qy": float(waypoints[i, 5]),
            "qz": float(waypoints[i, 6]),
            "zone": zp.source,
            "pzone_tcp_mm": zp.pzone_tcp_mm,
            "r_eff_mm": zp.eff_pzone_tcp_mm,
            "path_deviation_max_mm": dev_max,
            "v_cmd_mm_s": float(v_cmd[i]),
        })

    fig.add_trace(go.Scatter3d(
        x=wp_mm[:, 0], y=wp_mm[:, 1], z=wp_mm[:, 2],
        mode="markers+text",
        marker=dict(color="green", size=5, symbol="diamond"),
        text=[str(i) for i in range(n_wp)],
        textposition="top center",
        textfont=dict(size=8),
        hovertext=hover_texts,
        hoverinfo="text",
        name="Waypoints",
    ))

    # Orientation axes at waypoints (every Nth)
    step = max(1, n_wp // 20)
    for i in range(0, n_wp, step):
        origin = wp_mm[i]
        ax_x, ax_y, ax_z = _quaternion_to_axes(waypoints[i, 3:7], scale=3.0)
        for ax_vec, color, label in [(ax_x, "red", "X"), (ax_y, "green", "Y"), (ax_z, "blue", "Z")]:
            end = origin + ax_vec
            fig.add_trace(go.Scatter3d(
                x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]],
                mode="lines", line=dict(color=color, width=2),
                showlegend=False, hoverinfo="skip",
            ))

    fig.update_layout(
        title=f"Zone Segments — Trajectory {traj_index} ({Path(csv_path).stem})",
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        width=1200, height=800,
    )

    html_path = out / f"zone_segments_traj{traj_index}.html"
    fig.write_html(str(html_path))
    print(f"Interactive HTML: {html_path}")

    # Static image
    try:
        fig.write_image(str(out / f"zone_segments_traj{traj_index}.png"),
                        width=1200, height=800, scale=2)
        print(f"Static PNG saved")
    except Exception:
        print("(kaleido not available — skipping PNG export)")

    # Zone summary JSON
    summary = {
        "source_file": csv_path,
        "trajectory_index": traj_index,
        "n_waypoints": n_wp,
        "n_dense_samples": dense_path.n_samples,
        "total_arc_length_mm": dense_path.total_arc_length_mm,
        "n_blend_arcs": sum(1 for g in blend_geoms if g is not None),
        "waypoints": zone_summary,
    }
    json_path = out / f"zone_summary_traj{traj_index}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Zone summary: {json_path}")

    # Also save a static matplotlib version
    _plot_matplotlib(wp_mm, dense_mm, dense_path, zones, blend_geoms,
                     out / f"zone_segments_traj{traj_index}_static.png",
                     traj_index, Path(csv_path).stem)


def _plot_matplotlib(
    wp_mm: np.ndarray,
    dense_mm: np.ndarray,
    dense_path,
    zones: List[ZoneParams],
    blend_geoms: list,
    out_path: Path,
    traj_idx: int,
    stem: str,
):
    """Fallback static matplotlib 3D plot."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    straight = ~dense_path.is_blend_arc
    blend = dense_path.is_blend_arc

    ax.plot(dense_mm[straight, 0], dense_mm[straight, 1], dense_mm[straight, 2],
            "b-", lw=1.5, alpha=0.7, label="Straight")
    if np.any(blend):
        ax.scatter(dense_mm[blend, 0], dense_mm[blend, 1], dense_mm[blend, 2],
                   c="orange", s=1, alpha=0.5, label="Blend arc")

    ax.scatter(wp_mm[:, 0], wp_mm[:, 1], wp_mm[:, 2],
               c="green", s=30, marker="^", zorder=5, label="Waypoints")

    for i, zp in enumerate(zones):
        if not zp.finep and zp.eff_pzone_tcp_mm > 0.1:
            ax.text(wp_mm[i, 0], wp_mm[i, 1], wp_mm[i, 2],
                    f"  r={zp.eff_pzone_tcp_mm:.1f}", fontsize=6, alpha=0.7)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Zone Segments — Traj {traj_idx} ({stem})")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Static matplotlib: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D zone segment visualization",
    )
    parser.add_argument("--csv", required=True, help="Toolpath CSV with zone data")
    parser.add_argument("--output", "-o", default="output/zone_segments",
                        help="Output directory")
    parser.add_argument("--base_frame", action="store_true",
                        help="Waypoints are in robot base frame (skip knife transform)")
    parser.add_argument("--knife", default="Zund", help="Knife pose name")
    parser.add_argument("--traj", type=int, default=0, help="Trajectory index (0-based)")
    parser.add_argument("--ds", type=float, default=0.5,
                        help="Dense path sampling resolution (mm)")
    args = parser.parse_args()

    generate_zone_visualization(
        csv_path=args.csv,
        output_dir=args.output,
        base_frame=args.base_frame,
        knife_name=args.knife,
        traj_index=args.traj,
        ds_mm=args.ds,
    )


if __name__ == "__main__":
    main()
