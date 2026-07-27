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

    # also load the task's scene and render a .jpg per viewpoint (sensor_0 +
    # every generated angle) for a visual sanity check, instead of trusting
    # the numbers blind -- saved under camera_check/gen_camera_poses/<task_name>/
    python scripts/gen_camera_poses.py --position ... --orientation ... \\
        --angles 60 -60 --task_name nav_to_table_02

Every run also appends a timestamped record (all poses it produced, plus the
CLI args used) as one line of JSON to --log_path (default:
camera_check/gen_camera_poses/poses_log.jsonl) -- so past runs (e.g. the
original --angles 60 -60 pass, then a later --nudge_forward tweak of just
sensor_2) stay on record instead of only living in scrollback.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

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


def render_camera_poses(task_name: str, poses: list, render_dir: str, headless: bool = False) -> None:
    """Load *task_name*'s scene and save one .jpg per (label, position, orientation)
    in *poses*, so generated viewpoints can be checked visually instead of trusting
    the numbers blind. Heavy imports are local to this function so the default
    pure-math CLI path stays fast and dependency-light.
    """
    import os

    import numpy as np
    import torch as th
    from PIL import Image

    import omnigibson as og
    from omnigibson.macros import gm

    from damagesim.omnigibson.damageable_env import OGDamageableEnvironment
    from scripts.teleop_b1k import TASK_REGISTRY, load_task_config, build_env_config

    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")

    os.makedirs(render_dir, exist_ok=True)

    # Also settable via OMNIGIBSON_HEADLESS=true (macros.py reads that env var
    # at import time); --headless just avoids having to export it separately.
    if headless:
        gm.HEADLESS = True

    task_cfg, task_mod = load_task_config(task_name)
    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    env_config = build_env_config(task_cfg)
    env = OGDamageableEnvironment(configs=env_config)
    env.reset()
    if hasattr(task_mod, "reset") and callable(task_mod.reset):
        try:
            task_mod.reset(env)
        except Exception as e:
            print(f"[gen_camera_poses] task_mod.reset(env) raised, continuing with base reset: {e}")

    try:
        for label, pos, orn in poses:
            og.sim.viewer_camera.set_position_orientation(
                position=th.tensor(pos, dtype=th.float32),
                orientation=th.tensor(orn, dtype=th.float32),
            )
            for _ in range(5):
                og.sim.render()

            obs, _ = og.sim.viewer_camera.get_obs()
            frame = obs["rgb"]
            frame = frame.cpu().numpy() if isinstance(frame, th.Tensor) else np.array(frame)
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

            fpath = os.path.join(render_dir, f"{label}.jpg")
            Image.fromarray(frame).save(fpath, quality=95)
            print(f"[gen_camera_poses] saved {fpath}")
    finally:
        og.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--position", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--orientation", type=float, nargs=4, required=True, metavar=("X", "Y", "Z", "W"))
    p.add_argument("--angles", type=float, nargs="+", default=[],
                   help="Degrees to rotate around the vertical axis through the look-at target, "
                        "one new camera per angle (e.g. --angles 70 -70). Optional -- omit to "
                        "just nudge/inspect a single --position/--orientation with no rotated copies.")
    p.add_argument("--nudge_forward", type=float, default=0.0,
                   help="Translate --position this many meters along its OWN view direction "
                        "(camera-local -Z) before doing anything else -- e.g. re-paste an "
                        "already-generated external_sensor_2's position/orientation here and "
                        "set --nudge_forward 0.3 to push it forward, instead of re-deriving the "
                        "whole rotate-around-target set from scratch.")
    p.add_argument("--nudge_right", type=float, default=0.0,
                   help="Translate --position along its own local +X (right) axis (meters).")
    p.add_argument("--nudge_up", type=float, default=0.0,
                   help="Translate --position along its own local +Y (up) axis (meters).")
    p.add_argument("--label", type=str, default="external_sensor_0",
                   help="Name to print/render the given --position/--orientation (post-nudge) "
                        "under, e.g. --label external_sensor_2 when nudging an already-generated "
                        "sensor_2 rather than the primary sensor_0 (default: external_sensor_0).")
    p.add_argument("--target_z", type=float, default=0.75,
                   help="Assumed workspace height (m) for inferring the look-at target "
                        "when --target is not given.")
    p.add_argument("--target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Explicit look-at target, overrides ray/--target_z inference.")
    p.add_argument("--horizontal_aperture", type=float, default=15.0)
    p.add_argument("--start_index", type=int, default=1,
                   help="First external_sensor_N index to assign to generated cameras.")
    p.add_argument("--task_name", type=str, default=None,
                   help="If given, also loads this task's scene and renders/saves a .jpg "
                        "for external_sensor_0 (the original pose) and every generated "
                        "viewpoint, for a visual sanity check (see scripts/teleop_b1k.py's "
                        "TASK_REGISTRY for valid names).")
    p.add_argument("--render_dir", type=str, default=None,
                   help="Directory to save rendered .jpg per camera pose into (default: "
                        "camera_check/gen_camera_poses/<task_name>/). Only used with --task_name.")
    p.add_argument("--headless", action="store_true",
                   help="Run the render pass headless (no GUI window). Only used with "
                        "--task_name; equivalent to OMNIGIBSON_HEADLESS=true.")
    p.add_argument("--log_path", type=str,
                   default=os.path.join("camera_check", "gen_camera_poses", "poses_log.jsonl"),
                   help="Append a timestamped JSON-lines record of this run's poses + args here "
                        "(default: camera_check/gen_camera_poses/poses_log.jsonl). Pass an empty "
                        "string to disable logging.")
    args = p.parse_args()

    position = np.array(args.position, dtype=np.float64)
    orientation = np.array(args.orientation, dtype=np.float64)
    R0 = quat_to_mat(orientation)

    if args.nudge_forward or args.nudge_right or args.nudge_up:
        forward_w, right_w, up_w = -R0[:, 2], R0[:, 0], R0[:, 1]
        position = position + (
            forward_w * args.nudge_forward + right_w * args.nudge_right + up_w * args.nudge_up
        )
        print(f"# nudged position (forward={args.nudge_forward}, right={args.nudge_right}, "
              f"up={args.nudge_up}): {position.tolist()}")

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

    if args.nudge_forward or args.nudge_right or args.nudge_up:
        # Orientation is unchanged by a nudge -- print the updated block too
        # (under --label, e.g. external_sensor_2) so the new position can be
        # copied straight back into the task file.
        print(f'    "{args.label}": {{')
        print(f'        "position": [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}],')
        print(f'        "orientation": [{orientation[0]:.4f}, {orientation[1]:.4f}, '
              f'{orientation[2]:.4f}, {orientation[3]:.4f}],')
        print(f'        "horizontal_aperture": {args.horizontal_aperture},')
        print('    },')

    radius_vec = position - target
    poses = [(args.label, position.tolist(), orientation.tolist())]
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
        poses.append((f"external_sensor_{idx}", new_pos.tolist(), new_quat.tolist()))

    if args.log_path:
        # Written BEFORE the render pass below on purpose: og.shutdown() at
        # the end of render_camera_poses() (Isaac Sim app teardown) can
        # terminate the process abruptly and never return control here, so
        # logging after rendering was silently dropping records.
        log_path = args.log_path
        if os.path.isdir(log_path):
            # Common slip: pointing --log_path at a directory (e.g. reusing
            # --render_dir) instead of a file. Fall back to the default
            # filename inside it rather than crashing on open().
            fixed = os.path.join(log_path, "poses_log.jsonl")
            print(f"# --log_path '{log_path}' is a directory, not a file -- "
                  f"writing to {fixed} instead")
            log_path = fixed

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "task_name": args.task_name,
            "args": {k: v for k, v in vars(args).items() if k != "log_path"},
            "poses": {
                label: {"position": pos, "orientation": orn}
                for label, pos, orn in poses
            },
        }
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"\n# appended run record to {log_path}")

    if args.task_name is not None:
        render_dir = args.render_dir or os.path.join("camera_check", "gen_camera_poses", args.task_name)
        print(f"\n# rendering {len(poses)} viewpoint(s) for task '{args.task_name}' into {render_dir}/ ...")
        render_camera_poses(args.task_name, poses, render_dir, headless=args.headless)


if __name__ == "__main__":
    main()
