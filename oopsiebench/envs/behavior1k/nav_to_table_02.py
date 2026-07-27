"""
Task configuration for **nav_to_table_02**.

Scene : Pomaria_1_int (kitchen_0's own breakfast_table as the obstacle)
Robot : Tiago (tiago0)
Damage: mechanical (plate + vase are fragile decor on top of the obstacle table)

Unlike ``nav_to_table`` (Rs_int, injected pedestal_table + injected fragile
props), this variant reuses the scene's *native* breakfast_table as the
obstacle the robot must navigate around, and adds a plate + a vase on top of
it as the fragile props. The plate is also the pick/carry target (mirrors
``nav_to_table``'s water_bottle lift-check); the vase is decor-only and is
just health/force-monitored, not picked.

No init_states pkl is authored yet (``INIT_STATE_PATH = None``) — reset()
seats the plate/vase on the table programmatically every episode, same
pattern as open_single_door / pick_egg. The robot start position below is a
first guess (3m from the table, opposite of the intended approach direction)
and will likely need interactive tuning in teleop_b1k.py to avoid spawning
inside a wall/other furniture, since exact room polygons weren't available
offline.
"""

from __future__ import annotations

import numpy as np
import omnigibson as og
import torch as th
from omnigibson.controllers.controller_base import IsGraspingState
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "tiago0"
ROBOT_TYPE = "Tiago"

INIT_STATE_PATH = None

# ── Task objects ─────────────────────────────────────────────────────────

# breakfast_table_uhrsex_0 (native to Pomaria_1_int, kitchen_0) sits at
# world xy ~= (-0.41, -1.96). Plate/vase spawn above it; reset() seats them
# onto its top surface via runtime AABB lookup (same helper as pick_egg).
_TABLE_XY = (-0.4119, -1.9556)

TASK_OBJECTS = {
    "plate": {
        "type": "DatasetObject",
        "name": "plate",
        "category": "plate",
        "model": "ntedfx",  # same model used by place_plate
        "position": [_TABLE_XY[0] + 0.15, _TABLE_XY[1] - 0.15, 1.5],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "vase": {
        "type": "DatasetObject",
        "name": "vase",
        "category": "vase",
        "model": "uuypot",  # same model used by nav_to_table
        "position": [_TABLE_XY[0] - 0.15, _TABLE_XY[1] + 0.15, 1.5],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 0.5],
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────
# Placeholder viewpoint looking at the table; needs interactive tuning
# alongside the robot spawn position once the scene is loaded.
VIEWER_CAMERA_POS = [-0.4166698455810547, 0.061550360172986984, 1.8746356964111328]
VIEWER_CAMERA_ORN = [3.671550075617872e-33, 0.543687105178833, 0.8392879962921143, 6.123234262925839e-17]


EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": [1.0462780215591951, -1.6044410903402793, 1.5067079013892717],
        "orientation": [0.2718435530927206, 0.4708468456666398, 0.7268447206424323, 0.41964399512196665],
        "horizontal_aperture": 20.995,
        "world_fixed": True,
    },
    "external_sensor_1": {
        "position": [-2.5909346977833536, -1.193762037535552, 1.8746356964111328],
        "orientation": [-0.2718435530927206, 0.4708468456666398, 0.7268447206424323, -0.41964399512196665],
        "horizontal_aperture": 20.995,
        "world_fixed": True,
    },
}

_U_XY = 0.03
_U_YAW = 0.12
_LIFT_Z = 0.1


def _support_surface_top_z(env, obj) -> float:
    """Top-z of the nearest surface directly beneath obj's xy (e.g. the table)."""
    obj_pos, _ = obj.get_position_orientation()
    obj_pos = obj_pos if isinstance(obj_pos, th.Tensor) else th.tensor(obj_pos, dtype=th.float32)
    target_z = float(obj_pos[2])

    def _inside_xy(p, amin, amax):
        return (float(amin[0]) <= float(p[0]) <= float(amax[0])
                and float(amin[1]) <= float(p[1]) <= float(amax[1]))

    excluded = {obj.name}
    for r in getattr(env, "robots", []) or []:
        if hasattr(r, "name"):
            excluded.add(r.name)

    candidates = []
    for scene_obj in getattr(env.scene, "objects", []) or []:
        if scene_obj is None or getattr(scene_obj, "name", None) in excluded:
            continue
        if not hasattr(scene_obj, "aabb") or scene_obj.aabb is None:
            continue
        amin, amax = scene_obj.aabb
        top_z = float(amax[2])
        if top_z >= target_z or not _inside_xy(obj_pos, amin, amax):
            continue
        candidates.append((target_z - top_z, top_z))

    if not candidates:
        raise RuntimeError(f"[nav_to_table_02] no supporting surface found under {obj.name}")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _seat_on_table(env, obj):
    obj_pos, obj_orn = obj.get_position_orientation()
    top_z = _support_surface_top_z(env, obj)
    aabb = getattr(obj, "aabb", None)
    half_h = 0.5 * float(aabb[1][2] - aabb[0][2]) if aabb is not None else 0.0
    pos = obj_pos.clone()
    pos[2] = top_z + half_h + 0.002
    obj.set_position_orientation(pos, obj_orn)
    obj.keep_still()


def reset(env):
    """Seat plate + vase on the breakfast_table; jitter the Tiago base; settle."""
    if not getattr(env, "robots", None):
        return
    robot = env.robots[0]

    for _ in range(5):
        og.sim.step()

    for name in ("plate", "vase"):
        obj = env.scene.object_registry("name", name)
        if obj is None:
            continue
        try:
            _seat_on_table(env, obj)
        except RuntimeError as e:
            print(e)

    pos, orn = robot.get_position_orientation()
    pos = pos.clone()
    pos[0] += float(np.random.uniform(-_U_XY, _U_XY))
    pos[1] += float(np.random.uniform(-_U_XY, _U_XY))
    euler = T.quat2euler(orn).clone()
    euler[2] = euler[2] + float(np.random.uniform(-_U_YAW, _U_YAW))
    robot.set_position_orientation(pos, T.euler2quat(euler))
    robot.set_joint_velocities(th.zeros(robot.n_dof, device=pos.device, dtype=pos.dtype))
    robot.keep_still()

    for _ in range(10):
        og.sim.step()

    plate = env.scene.object_registry("name", "plate")
    if plate is not None:
        p, _ = plate.get_position_orientation()
        env._nav02_plate_start_z = float(p[2])
    else:
        env._nav02_plate_start_z = None


def playback_reset(env):
    """reset() isn't run during playback — re-seat plate/vase so they start on the table."""
    for name in ("plate", "vase"):
        obj = env.scene.object_registry("name", name)
        if obj is None:
            continue
        try:
            _seat_on_table(env, obj)
        except RuntimeError as e:
            print(e)


def task_completion_check(env):
    start_z = getattr(env, "_nav02_plate_start_z", None)
    if start_z is None:
        return False
    plate = env.scene.object_registry("name", "plate")
    if plate is None or not getattr(env, "robots", None):
        return False
    plate_pos, _ = plate.get_position_orientation()
    lifted = (float(plate_pos[2]) - start_z) >= _LIFT_Z
    robot_grasping = env.robots[0].is_grasping(candidate_obj=plate).value == IsGraspingState.TRUE
    return lifted and robot_grasping


def register_teleop_keys(env, kb):
    """Teleop post-setup hook: show the robot's head camera in a docked side viewport."""
    import omnigibson.lazy as lazy
    from omnigibson.sensors import VisionSensor
    from omnigibson.utils.ui_utils import dock_window

    try:
        cam = next((s for s in env.robots[0].sensors.values()
                    if isinstance(s, VisionSensor)), None)
        if cam is None:
            print("[nav_to_table_02] no robot camera found; skipping head-camera viewport")
            return
        cam.viewer_visibility = True
        dock_window(
            space=lazy.omni.ui.Workspace.get_window("DockSpace"),
            name=cam._viewport.name,
            location=lazy.omni.ui.DockPosition.LEFT,
            ratio=0.3,
        )
        for _ in range(5):
            og.sim.render()
        print(f"[nav_to_table_02] head camera '{cam.name}' shown in '{cam._viewport.name}'")
    except Exception as e:
        print(f"[nav_to_table_02] could not show head-camera viewport: {e}")


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="nav_to_table_02",

        # No fluids/cloth in this task — CPU physics (GPU dynamics inherited from
        # nav_to_table's config is unnecessary here; see door_and_mug.py for the
        # PhysX CUDA error 700 it can trigger on scene load).
        use_gpu_dynamics=False,
        enable_transition_rules=False,

        scene_config={
            "type": "InteractiveTraversableScene",
            "scene_model": "Pomaria_1_int",
            "include_robots": False,
            "load_object_categories": ["floors", "walls", "breakfast_table"],
            "load_room_instances": ["kitchen_0", "corridor_0"],
        },

        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            # First-guess spawn 3m south of the table in kitchen_0; needs
            # interactive verification (walls / other furniture not modeled here).
            "position": [-0.41190001368522644, -0.5379675030708313, 0.00034388541826047003],
            "orientation": [1.07123510062479e-08, -2.5411512183382e-07, -0.7071067094802856, 0.7071067094802856],
            "default_arm_pose": "horizontal",
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth"],
            "action_normalize": False,
            "self_collisions": True,
            "controller_config": {
                "arm_left": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_left": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
                "arm_right": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_right": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
            },
            "exclude_sensor_names": ["left_eef_link", "right_eef_link"],
        },

        task_objects=TASK_OBJECTS,

        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,

        target_objects_health_with_links=[
            "plate@base_link",
            "vase@base_link",
        ],
        target_objects_health=["plate", "vase"],
        target_objects_forces=[
            "plate@base_link",
            "vase@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],

        default_collect_hdf5="demos/behavior1k/teleop_data/nav_to_table_02.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/nav_to_table_02_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/nav_to_table_02",
    )
