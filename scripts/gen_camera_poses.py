"""
Generates additional EXTERNAL_CAMERA_CONFIGS entries for a scenario, by
rotating an existing camera around a vertical axis through its inferred
look-at target (same height/distance as the source camera).

OG/USD cameras use OpenGL convention: look down -Z, +Y up (see
damagesim/omnigibson/pointworld_export.py's _opengl_pose_to_cv_world2cam
docstring). Given an existing camera pose (position + xyzw quaternion), this
script:

  1. Infers a look-at target by intersecting the camera's forward ray with
     the horizontal plane z = --target_z (default 0.75m, a typical
     counter/table height).
  2. Rotates the camera position around that target's vertical axis by each
     angle in --angles, keeping height and radius fixed.
  3. Computes a roll-free look-at orientation (xyzw) for each new position.
  4. Self-checks by re-deriving a look-at quaternion for the ORIGINAL camera
     from its own inferred target, and reports the geodesic angle to the
     original orientation -- a large angle means --target_z (or an explicit
     --target) is a bad guess for this scene and should be corrected before
     trusting the generated poses.

Usage:
    python scripts/gen_camera_poses.py \\
        --position 7.392 -0.6436 1.7519 \\
        --orientation 0.5273 0.2970 0.3907 0.6936 \\
        --angles 70 -70

    # override the inferred look-at target explicitly (x y z)
    python scripts/gen_camera_poses.py --position ... --orientation ... \\
        --target 4.06 -0.26 0.94 --angles 75
"""

from __future__ import annotations

import argparse

import numpy as np


def quat_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def rotz(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def infer_target(position: np.ndarray, R: np.ndarray, target_z: float) -> np.ndarray:
    """Intersect the camera's forward ray (-Z column of R, OpenGL convention)
    with the horizontal plane z = target_z."""
    forward = -R[:, 2]
    if abs(forward[2]) < 0.05:
        raise ValueError(
            f"Camera forward ray is nearly horizontal (forward.z={forward[2]:.3f}); "
            "cannot infer a look-at target on a horizontal plane -- pass --target explicitly."
        )
    s = (target_z - position[2]) / forward[2]
    if s <= 0:
        raise ValueError(
            f"Inferred target is behind the camera (s={s:.3f}); check --target_z "
            "or pass --target explicitly."
        )
    return position + s * forward


def look_at_quat(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Roll-free look-at orientation (xyzw) for an OpenGL-convention camera
    (looks down -Z, +Y up)."""
    z_c = position - target  # camera -Z axis points toward target -> +Z away from it
    z_c = z_c / np.linalg.norm(z_c)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_c, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    x_c = np.cross(world_up, z_c)
    x_c = x_c / np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    R = np.column_stack([x_c, y_c, z_c])
    return mat_to_quat(R)


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    dot = abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--orientation", type=float, nargs=4, required=True, metavar=("X", "Y", "Z", "W"))
    p.add_argument("--angles", type=float, nargs="+", required=True,
                   help="Degrees to rotate around the vertical axis through the look-at target, "
                        "one new camera per angle (e.g. --angles 70 -70).")
    p.add_argument("--target_z", type=float, default=0.75,
                   help="Assumed workspace height (m) for inferring the look-at target "
                        "when --target is not given.")
    p.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Explicit look-at target, overrides ray/--target_z inference.")
    p.add_argument("--horizontal_aperture", type=float, default=15.0)
    p.add_argument("--start_index", type=int, default=1,
                   help="First external_sensor_N index to assign to generated cameras.")
    args = p.parse_args()

    position = np.array(args.position, dtype=np.float64)
    orientation = np.array(args.orientation, dtype=np.float64)
    R0 = quat_to_mat(orientation)

    if args.target is not None:
        target = np.array(args.target, dtype=np.float64)
    else:
        target = infer_target(position, R0, args.target_z)

    # Self-check: re-derive a look-at quat for the ORIGINAL camera from the
    # inferred target and compare to its actual orientation.
    q_check = look_at_quat(position, target)
    angle_err = quat_angle_deg(q_check, orientation)
    print(f"# inferred target: {target.tolist()}")
    print(f"# self-check angle error vs original orientation: {angle_err:.1f} deg", end="")
    if angle_err > 10.0:
        print("  <-- WARNING: >10 deg, target_z/--target is likely a poor fit for this scene")
    else:
        print("  (ok)")
    print()

    radius_vec = position - target
    for i, angle_deg in enumerate(args.angles):
        theta = np.radians(angle_deg)
        new_pos = target + rotz(theta) @ radius_vec
        new_quat = look_at_quat(new_pos, target)
        idx = args.start_index + i
        print(f'    "external_sensor_{idx}": {{')
        print(f'        "position": [{new_pos[0]:.4f}, {new_pos[1]:.4f}, {new_pos[2]:.4f}],')
        print(f'        "orientation": [{new_quat[0]:.4f}, {new_quat[1]:.4f}, '
              f'{new_quat[2]:.4f}, {new_quat[3]:.4f}],')
        print(f'        "horizontal_aperture": {args.horizontal_aperture},')
        print('    },')


if __name__ == "__main__":
    main()
