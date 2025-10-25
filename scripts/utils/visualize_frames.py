# Python code to visualize the origin frame and the given frame
# (translation in mm and quaternion w,x,y,z).
# It draws RGB arrows for X (red), Y (green), Z (blue) for both frames.
# The script will display the figure.
# Requirements: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import xml.etree.ElementTree as ET
from math_utils import quat_to_rot_matrix, rpy_to_rot_matrix, make_transform


def parse_urdf(urdf_path):
    """Parse a URDF file and return links, joints, and a parent->joints map.
    Returns:
      links: set of link names
      joints: list of dicts {name,type,parent,child,origin_xyz,origin_rpy,axis}
      children_by_parent: dict parent_link -> list(joint)
      root_link: inferred root link name (or first if ambiguous)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = set()
    for l in root.findall('link'):
        name = l.get('name')
        if name:
            links.add(name)

    joints = []
    child_links = set()
    for j in root.findall('joint'):
        jname = j.get('name') or ''
        jtype = j.get('type') or 'fixed'
        parent = j.find('parent').get('link')
        child = j.find('child').get('link')
        child_links.add(child)

        origin = j.find('origin')
        if origin is not None:
            xyz_str = origin.get('xyz') or '0 0 0'
            rpy_str = origin.get('rpy') or '0 0 0'
        else:
            xyz_str = '0 0 0'
            rpy_str = '0 0 0'
        # Convert from meters (URDF) to millimeters
        origin_xyz = np.fromstring(xyz_str, sep=' ') * 1000.0  # m to mm
        origin_rpy = np.fromstring(rpy_str, sep=' ')  # radians, no conversion needed

        axis_el = j.find('axis')
        if axis_el is not None and axis_el.get('xyz') is not None:
            axis = np.fromstring(axis_el.get('xyz'), sep=' ')
        else:
            axis = np.array([1.0, 0.0, 0.0])  # URDF default

        joints.append({
            'name': jname,
            'type': jtype,
            'parent': parent,
            'child': child,
            'origin_xyz': origin_xyz,
            'origin_rpy': origin_rpy,
            'axis': axis
        })

    # Build adjacency
    children_by_parent = {}
    for jd in joints:
        children_by_parent.setdefault(jd['parent'], []).append(jd)

    # Infer root: link that is never a child
    roots = list(links - child_links)
    root_link = roots[0] if roots else (next(iter(links)) if links else None)

    return links, joints, children_by_parent, root_link

def draw_frame(ax, T, axis_len, linewidth=2.0, alpha=1.0, label=None):
    """Draw RGB axes for transform T (4x4). Optionally add text label."""
    R = T[:3, :3]
    p = T[:3, 3]
    ax.quiver(p[0], p[1], p[2], R[0, 0]*axis_len, R[1, 0]*axis_len,
              R[2, 0]*axis_len, color='r', linewidth=linewidth,
              arrow_length_ratio=0.1, alpha=alpha)
    ax.quiver(p[0], p[1], p[2], R[0, 1]*axis_len, R[1, 1]*axis_len,
              R[2, 1]*axis_len, color='g', linewidth=linewidth,
              arrow_length_ratio=0.1, alpha=alpha)
    ax.quiver(p[0], p[1], p[2], R[0, 2]*axis_len, R[1, 2]*axis_len,
              R[2, 2]*axis_len, color='b', linewidth=linewidth,
              arrow_length_ratio=0.1, alpha=alpha)

    # Add text label if provided
    if label is not None:
        # Offset the label slightly to avoid overlapping with arrows
        label_offset = axis_len * 0.3
        ax.text(p[0] + label_offset, p[1] + label_offset, p[2] + label_offset,
                label, color='k', fontsize=8, alpha=0.8)

    return np.vstack([p, p + R[:, 0]*axis_len, p + R[:, 1]*axis_len,
                      p + R[:, 2]*axis_len])

def draw_urdf_model(ax, urdf_path, base_link=None,
                     joint_axis_len=120.0, frame_axis_len=140.0,
                     text=False):
    """Draw all link frames and joint axes from a URDF on the given axes.
    Returns an (M,3) array of points used for autoscaling.

    Args:
        ax: matplotlib 3D axis
        urdf_path: path to URDF file
        base_link: optional base link name
        joint_axis_len: length of joint axis arrows
        frame_axis_len: length of frame coordinate arrows
        text: whether to show joint name labels
    """
    links, joints, children_by_parent, inferred_root = parse_urdf(urdf_path)
    root_link = base_link or inferred_root
    if root_link is None:
        return np.zeros((0,3))

    # BFS through kinematic tree assuming zero joint positions
    T_world = {root_link: np.eye(4)}
    points = []

    from collections import deque
    q = deque([root_link])
    while q:
        parent = q.popleft()
        T_parent = T_world[parent]

        # Draw parent link frame
        points.append(draw_frame(ax, T_parent, frame_axis_len,
                                 linewidth=1.5, alpha=0.9))

        for jd in children_by_parent.get(parent, []):
            # Joint frame relative to parent
            T_pj = make_transform(jd['origin_xyz'], jd['origin_rpy'])
            T_joint = T_parent @ T_pj

            # Draw joint axis in world
            R_joint = T_joint[:3, :3]
            p_joint = T_joint[:3, 3]
            axis_dir = R_joint @ (jd['axis'] / (np.linalg.norm(jd['axis']) + 1e-12))

            # Draw simple axis arrow for all joint types (keeping it simple)
            ax.quiver(p_joint[0], p_joint[1], p_joint[2],
                      axis_dir[0]*joint_axis_len, axis_dir[1]*joint_axis_len,
                      axis_dir[2]*joint_axis_len, color='k', linewidth=1.5,
                      arrow_length_ratio=0.1, alpha=0.7)

            # Zero joint motion (identity)
            T_child = T_joint
            T_world[jd['child']] = T_child
            q.append(jd['child'])

            # Draw child link frame with joint name as label
            points.append(draw_frame(ax, T_child, frame_axis_len,
                                     linewidth=1.5, alpha=0.9,
                                     label=jd['name']))

            # Connector line between links (thicker and more visible)
            ax.plot([T_parent[0,3], T_child[0,3]], [T_parent[1,3], T_child[1,3]], [T_parent[2,3], T_child[2,3]],
                    color='gray', linewidth=2.0, alpha=0.8)

    if points:
        return np.vstack(points)
    return np.zeros((0,3))

def set_axes_equal(ax):
    """Make 3D axes equal scale (workaround for matplotlib)."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# CLI to optionally load a URDF and/or visualize a single frame
parser = argparse.ArgumentParser(
    description='Visualize origin, a transformed frame, and optionally '
                'a URDF robot model.')
parser.add_argument('--urdf', type=str, default=None,
                    help='Path to URDF file to visualize robot frames '
                         'and joint axes')
parser.add_argument('--base-link', type=str, default=None,
                    help='Base/root link name (optional)')
parser.add_argument('--tx', type=float, default=-367.773,
                    help='Tool/frame translation X (mm)')
parser.add_argument('--ty', type=float, default=-915.815,
                    help='Tool/frame translation Y (mm)')
parser.add_argument('--tz', type=float, default=520.4,
                    help='Tool/frame translation Z (mm)')
parser.add_argument('--qw', type=float, default=0.00515984,
                    help='Frame quaternion w')
parser.add_argument('--qx', type=float, default=0.712632,
                    help='Frame quaternion x')
parser.add_argument('--qy', type=float, default=-0.701518,
                    help='Frame quaternion y')
parser.add_argument('--qz', type=float, default=0.000396522,
                    help='Frame quaternion z')
parser.add_argument('--labels', action='store_true',
                    help='Show text labels for joint names')
args = parser.parse_args()

# --- Knife Tool (stationary) frame (from Jared's email) ---
translation_mm = np.array([args.tx, args.ty, args.tz])  # mm
quat_wxyz = np.array([args.qw, args.qx, args.qy, args.qz])  # w, x, y, z

# Build axes
origin = np.zeros(3)
unit_axes = np.eye(3)  # columns: x, y, z basis vectors
R = quat_to_rot_matrix(quat_wxyz)  # rotation matrix for the given quaternion
rotated_axes = R.dot(unit_axes)    # rotated basis vectors (columns)

# Plot settings
arrow_length = 70.0  # mm - controls arrow length visually (adjust if needed)
linewidth = 3.0
arrow_mutation_scale = 10  # for quiver arrow heads

fig = plt.figure(figsize=(11,9))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

# Plot origin frame axes (red, green, blue) with legend labels
x_arrow = ax.quiver(
    origin[0], origin[1], origin[2],
    unit_axes[0, 0]*arrow_length, unit_axes[1, 0]*arrow_length,
    unit_axes[2, 0]*arrow_length,
    color='r', linewidth=linewidth, arrow_length_ratio=0.1,
    label='X axis (Red)')
y_arrow = ax.quiver(
    origin[0], origin[1], origin[2],
    unit_axes[0, 1]*arrow_length, unit_axes[1, 1]*arrow_length,
    unit_axes[2, 1]*arrow_length,
    color='g', linewidth=linewidth, arrow_length_ratio=0.1,
    label='Y axis (Green)')
z_arrow = ax.quiver(
    origin[0], origin[1], origin[2],
    unit_axes[0, 2]*arrow_length, unit_axes[1, 2]*arrow_length,
    unit_axes[2, 2]*arrow_length,
    color='b', linewidth=linewidth, arrow_length_ratio=0.1,
    label='Z axis (Blue)')

# Plot transformed frame axes at translation (same RGB order)
ax.quiver(
    translation_mm[0], translation_mm[1], translation_mm[2],
    rotated_axes[0, 0]*arrow_length, rotated_axes[1, 0]*arrow_length,
    rotated_axes[2, 0]*arrow_length,
    color='r', linewidth=linewidth, arrow_length_ratio=0.1)
ax.quiver(
    translation_mm[0], translation_mm[1], translation_mm[2],
    rotated_axes[0, 1]*arrow_length, rotated_axes[1, 1]*arrow_length,
    rotated_axes[2, 1]*arrow_length,
    color='g', linewidth=linewidth, arrow_length_ratio=0.1)
ax.quiver(
    translation_mm[0], translation_mm[1], translation_mm[2],
    rotated_axes[0, 2]*arrow_length, rotated_axes[1, 2]*arrow_length,
    rotated_axes[2, 2]*arrow_length,
    color='b', linewidth=linewidth, arrow_length_ratio=0.1)

# Draw rectangle at Robot Base frame origin
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
rect_size = 20.0
rect_corners = np.array([
    [-rect_size, -rect_size, 0],
    [rect_size, -rect_size, 0],
    [rect_size, rect_size, 0],
    [-rect_size, rect_size, 0]
])
rect_faces = [rect_corners]
rect_3d = Poly3DCollection(rect_faces, alpha=0.3, facecolor='gray',
                           edgecolor='black', linewidth=2)
ax.add_collection3d(rect_3d)

# Origin markers and labels
ax.scatter(0,0,0, color='k', s=50, marker='s', label='Robot Base')
ax.text(30,30,30, 'Robot Base', color='k', fontsize=10, fontweight='bold')

# Add upward arrow for knife tool frame
up_arrow_length = 50.0
ax.quiver(
    translation_mm[0], translation_mm[1], translation_mm[2],
    0, 0, up_arrow_length,
    color='purple', linewidth=2.5, arrow_length_ratio=0.3, alpha=0.8,
    label='Tool Orientation')

ax.scatter(translation_mm[0], translation_mm[1], translation_mm[2],
           color='purple', s=50, marker='^')
ax.text(translation_mm[0] + 30, translation_mm[1] + 30,
        translation_mm[2] + 30, 'Knife tool', color='purple',
        fontsize=10, fontweight='bold')

# Annotate basis directions
label_offset = 25.0
ax.text(unit_axes[0, 0]*arrow_length + label_offset,
        unit_axes[1, 0]*arrow_length, unit_axes[2, 0]*arrow_length,
        'X_b', color='r')
ax.text(unit_axes[0, 1]*arrow_length,
        unit_axes[1, 1]*arrow_length + label_offset,
        unit_axes[2, 1]*arrow_length, 'Y_b', color='g')
ax.text(unit_axes[0, 2]*arrow_length,
        unit_axes[1, 2]*arrow_length,
        unit_axes[2, 2]*arrow_length + label_offset, 'Z_b', color='b')

ax.text(translation_mm[0] + rotated_axes[0, 0]*arrow_length + label_offset,
        translation_mm[1] + rotated_axes[1, 0]*arrow_length,
        translation_mm[2] + rotated_axes[2, 0]*arrow_length, 'X_k', color='r')
ax.text(translation_mm[0] + rotated_axes[0, 1]*arrow_length,
        translation_mm[1] + rotated_axes[1, 1]*arrow_length + label_offset,
        translation_mm[2] + rotated_axes[2, 1]*arrow_length, 'Y_k', color='g')
ax.text(translation_mm[0] + rotated_axes[0, 2]*arrow_length,
        translation_mm[1] + rotated_axes[1, 2]*arrow_length,
        translation_mm[2] + rotated_axes[2, 2]*arrow_length + label_offset,
        'Z_k', color='b')

# Optionally draw URDF robot model
urdf_points = np.zeros((0,3))
if args.urdf:
    try:
        urdf_points = draw_urdf_model(ax, args.urdf, base_link=args.base_link,
                                     joint_axis_len=120.0, frame_axis_len=100.0,
                                     text=args.labels)
    except Exception as e:
        print(f"Failed to load URDF '{args.urdf}': {e}")

# Axes labels and grid
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.grid(True)

# Make aspect equal
# first set ranges to include origin, frame, and URDF comfortably
frame_points = np.vstack([
    origin, translation_mm,
    origin + unit_axes.T*arrow_length,
    translation_mm + rotated_axes.T*arrow_length])
all_points = (frame_points if urdf_points.size == 0
              else np.vstack([frame_points, urdf_points]))
mins = all_points.min(axis=0) - 100
maxs = all_points.max(axis=0) + 100
ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.set_zlim(mins[2], maxs[2])
set_axes_equal(ax)

ax.view_init(elev=20, azim=-60)  # adjust view for a clear 3D perspective
plt.title('Origin frame (RGB) and Transformed frame at translation (RGB)')

# Add legend for axis colors
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)

# Save and show
plt.tight_layout()
plt.show()
