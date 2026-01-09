import yaml
import math
import random
import argparse
import os

def normalize_quaternion(q):
    """Normalize a quaternion to unit length."""
    w, x, y, z = q
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    return (w/norm, x/norm, y/norm, z/norm)


def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )


def generate_perturbed_quaternion(ref_quat, angle_deg, seed):
    """Generate a quaternion close to reference by small rotation."""
    random.seed(seed)
    
    # Random axis
    ax = random.gauss(0, 1)
    ay = random.gauss(0, 1)
    az = random.gauss(0, 1)
    axis_norm = math.sqrt(ax*ax + ay*ay + az*az)
    ax, ay, az = ax/axis_norm, ay/axis_norm, az/axis_norm
    
    # Small rotation quaternion
    angle_rad = math.radians(angle_deg)
    half_angle = angle_rad / 2
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)
    
    delta_q = (cos_half, ax*sin_half, ay*sin_half, az*sin_half)
    
    # Multiply quaternions
    result = quaternion_multiply(delta_q, ref_quat)
    return normalize_quaternion(result)


def linspace(start, stop, num):
    """Generate evenly spaced values."""
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]


def main():
    parser = argparse.ArgumentParser(description='Generate knife poses for feasibility analysis.')
    parser.add_argument('--num_x', type=int, default=5, help='Number of x-axis divisions (default: 5)')
    parser.add_argument('--num_y', type=int, default=5, help='Number of y-axis divisions (default: 5)')
    parser.add_argument('--num_z', type=int, default=3, help='Number of z-axis divisions (default: 3)')
    parser.add_argument('--num_ori', type=int, default=2, help='Number of orientation variations per grid point (default: 2)')
    parser.add_argument('--num_out_of_reach', type=int, default=6, help='Number of out-of-reach poses per robot (max 6, default: 6)')
    parser.add_argument('--output_path', type=str, default='config/generated_knife_poses.yaml', help='Path to save the generated poses (default: config/generated_knife_poses.yaml)')
    
    args = parser.parse_args()

    # Reference pose (from knife_config.yaml pose_1)
    ref_translation = (-367.773, -915.815, 520.4)
    ref_rotation = (0.00515984, 0.712632, -0.701518, 0.000396522)
    ref_rotation = normalize_quaternion(ref_rotation)
    
    poses = {}
    
    # 1. Generate nominal grid poses
    x_values = linspace(ref_translation[0] - 100, ref_translation[0] + 100, args.num_x)
    y_values = linspace(ref_translation[1] - 100, ref_translation[1] + 100, args.num_y)
    z_values = linspace(ref_translation[2] - 100, ref_translation[2] + 100, args.num_z)
    
    # Ensure all z values are positive
    z_values = [max(z, 1.0) for z in z_values]
    
    seed_counter = 42
    
    for x in x_values:
        for y in y_values:
            for z in z_values:
                # Generate orientations
                for i in range(args.num_ori):
                    ori_label = chr(65 + i)
                    pose_name = f"nominal_x{x:.1f}_y{y:.1f}_z{z:.1f}_ori{ori_label}"
                    
                    # Small perturbation for orientation
                    # Alternate between 5.0 and 8.0 as in original code
                    perturb_angle = 5.0 if i % 2 == 0 else 8.0
                    perturbed_quat = generate_perturbed_quaternion(ref_rotation, perturb_angle, seed_counter)
                    
                    seed_counter += 1
                    
                    poses[pose_name] = {
                        'description': 'nominal',
                        'intent': 'Nominal grid knife pose for workspace coverage',
                        'translation_mm': {
                            'x': x,
                            'y': y,
                            'z': z
                        },
                        'rotation': {
                            'w': perturbed_quat[0],
                            'x': perturbed_quat[1],
                            'y': perturbed_quat[2],
                            'z': perturbed_quat[3]
                        }
                    }
    
    # 2. Generate out-of-reach poses
    robot_configs = [
        (900, 950),
        (1150, 1200),
        (1400, 1450)
    ]
    
    sqrt2 = math.sqrt(2)
    full_directions = [
        ('+X', lambda r: (r, 0, 0)),
        ('-X', lambda r: (-r, 0, 0)),
        ('+Y', lambda r: (0, r, 0)),
        ('-Y', lambda r: (0, -r, 0)),
        ('diagonal', lambda r: (r/sqrt2, r/sqrt2, 0)),
        ('high_Z', lambda r: (0, 0, r))
    ]
    
    # Clip/cap the number of directions based on user input (max 6)
    num_out = max(0, min(len(full_directions), args.num_out_of_reach))
    directions = full_directions[:num_out]
    
    for reach_class, out_radius in robot_configs:
        for dir_name, pos_func in directions:
            offset = pos_func(out_radius)
            pose_name = f"out_of_reach_r{reach_class}_dir{dir_name}"
            
            poses[pose_name] = {
                'category': 'out_of_reach',
                'robot_reach_class': reach_class,
                'intent': f'Intentionally unreachable knife pose for reachability validation for robot {reach_class}',
                'translation_mm': {
                    'x': offset[0],
                    'y': offset[1],
                    'z': max(offset[2], 1.0)
                },
                'rotation': {
                    'w': ref_rotation[0],
                    'x': ref_rotation[1],
                    'y': ref_rotation[2],
                    'z': ref_rotation[3]
                }
            }
    
    # Write to YAML file
    output = {'poses': poses}
    
    output_path = args.output_path
    
    # Delete existing file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write the poses header
        f.write('poses:\n')
        
        # Write each pose with 2 blank lines between them
        pose_items = list(poses.items())
        for i, (pose_name, pose_data) in enumerate(pose_items):
            # Write the pose name
            f.write(f'  {pose_name}:\n')
            
            # Write pose data
            for key, value in pose_data.items():
                if isinstance(value, dict):
                    f.write(f'    {key}:\n')
                    for sub_key, sub_value in value.items():
                        f.write(f'      {sub_key}: {sub_value}\n')
                else:
                    f.write(f'    {key}: {value}\n')
            
            # Add 2 blank lines between poses (except for the last one)
            if i < len(pose_items) - 1:
                f.write('\n\n')
    
    print(f"Generated {len(poses)} knife poses")
    nominal_count = sum(1 for k in poses if k.startswith('nominal'))
    out_of_reach_count = sum(1 for k in poses if k.startswith('out_of_reach'))
    print(f"  - Nominal: {nominal_count}")
    print(f"  - Out-of-reach: {out_of_reach_count}")


if __name__ == '__main__':
    main()