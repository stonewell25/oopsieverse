"""
Task configuration for **door_slam_vase**.

Scene : Rs_int (Tiago primitives-style)
Robot : Tiago (tiago0)
Damage: mechanical

Sibling of nav_door_squeeze.py: same ``"door"`` category append (Rs_int's
own native door, if present in the loaded rooms, loads at its correct
dataset transform — zero new coordinates) and 100% identical TASK_OBJECTS/
robot/camera/pkl. The existing ``vase`` (already part of nav_to_table.py's
object set, sitting near the pedestal_table) *is* the object of interest
here — the difference from nav_door_squeeze.py is intended teleop narrative
(swing the base/arm near wherever the native door turns out to be, close to
the vase) rather than any code difference.
"""

import pickle

import numpy as np
import omnigibson as og
import torch as th

from oopsiebench.envs.behavior1k.base import TaskConfig, reset_randomize_enabled

ROBOT_NAME = "tiago0"
ROBOT_TYPE = "Tiago"

# ── Task objects ─────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "pedestal_table": {
        "type": "DatasetObject",
        "name": "pedestal_table",
        "category": "pedestal_table",
        "model": "djflkd",
        "position": [-0.5, 0.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 1.0],
    },
    "vase": {
        "type": "DatasetObject",
        "name": "vase",
        "category": "vase",
        "model": "uuypot",
        "position": [-0.5, -1.0, 0.10],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.5, 0.5, 0.5],
    },
    "swivel_chair": {
        "type": "DatasetObject",
        "name": "swivel_chair",
        "category": "swivel_chair",
        "model": "pkpcew",
        "position": [-0.5, 1.0, 0.50],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    "coffee_table": {
        "type": "DatasetObject",
        "name": "coffee_table",
        "category": "coffee_table",
        "model": "fqluyq",
        "position": [-0.4763, -1.2196, 0.2838],
        "orientation": [0.0, 0.0, 1.0, 0.0],
        "scale": [1.1712, 1.0266, 0.9481],
        "fixed_base": True,
    },
    "water_bottle": {
        "type": "DatasetObject",
        "name": "water_bottle",
        "category": "bottle_of_water",
        "model": "hrzznl",
        "position": [0.0, 0.0, 1.5],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.9, 0.9, 0.9],
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [1.5345655679702759, -2.3398592472076416, 1.3116816282272339]
VIEWER_CAMERA_ORN = [0.605172872543335, 0.14635765552520752, 0.18393288552761078, 0.7606010437011719]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 20.995,
        "world_fixed": True,
    },
    "external_sensor_1": {
        "position": [2.7406, -0.9182, 1.3117],
        "orientation": [0.5031, 0.3668, 0.4610, 0.6323],
        "horizontal_aperture": 20.995,
        "world_fixed": True,
    },
    "external_sensor_2": {
        "position": [-1.9813, -2.1673, 1.3117],
        "orientation": [0.5973, -0.1758, -0.2210, 0.7507],
        "horizontal_aperture": 20.995,
        "world_fixed": True,
    },
}

# ── Public entry point ───────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="door_slam_vase",
        use_gpu_dynamics=True,
        enable_transition_rules=False,
        scene_config={
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_object_categories": ["floors", "walls", "breakfast_table", "door"],
        },
        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
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
            "pedestal_table@base_link",
            "vase@base_link",
            "swivel_chair@base_link",
        ],
        target_objects_health=["pedestal_table", "vase", "swivel_chair"],
        target_objects_forces=[
            "pedestal_table@base_link",
            "vase@base_link",
            "swivel_chair@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],
        default_collect_hdf5="demos/behavior1k/teleop_data/door_slam_vase.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/door_slam_vase_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/door_slam_vase",
    )


_BOTTLE_U_XY = 0.03
_LIFT_Z = 0.1

INIT_STATE_PATH = "oopsiebench/envs/behavior1k/init_states/nav_to_table.pkl"


def reset(env):
    """Restore the saved init state, then snap the water bottle onto the table top."""
    with open(INIT_STATE_PATH, "rb") as f:
        state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)
    for _ in range(5):
        og.sim.step()

    try:
        bottle = env.scene.object_registry("name", "water_bottle")
        if bottle is None:
            return

        table = None
        for obj in getattr(env.scene, "objects", []) or []:
            name = (getattr(obj, "name", "") or "").lower()
            cat = (getattr(obj, "category", "") or "").lower()
            if "breakfast_table" in name or "breakfast_table" in cat:
                table = obj
                break
        if table is None:
            table = (env.scene.object_registry("category", "breakfast_table")
                     or env.scene.object_registry("name", "breakfast_table"))
        if table is None:
            return

        for _ in range(5):
            og.sim.step()

        if not hasattr(table, "aabb") or table.aabb is None:
            return
        tmin, tmax = table.aabb

        cx = (float(tmin[0]) + float(tmax[0])) * 0.5
        cy = (float(tmin[1]) + float(tmax[1])) * 0.5

        z_offset = 0.05
        if hasattr(bottle, "aabb") and bottle.aabb is not None:
            mmin, mmax = bottle.aabb
            z_offset = max(0.02, 0.5 * float(mmax[2] - mmin[2])) + 0.002

        randomize = reset_randomize_enabled()
        pos = th.tensor(
            [
                cx + (float(np.random.uniform(-_BOTTLE_U_XY, _BOTTLE_U_XY)) if randomize else 0.0),
                cy + (float(np.random.uniform(-_BOTTLE_U_XY, _BOTTLE_U_XY)) if randomize else 0.0),
                float(tmax[2]) + float(z_offset),
            ],
            dtype=th.float32,
        )
        bottle.set_position_orientation(pos, th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32))
        try:
            bottle.keep_still()
        except Exception:
            pass
        bottle_pos, _ = bottle.get_position_orientation()
        env._nav_bottle_start_z = float(bottle_pos[2])
    except Exception:
        return


def playback_reset(env):
    """reset() isn't called during playback — re-apply it so objects start arranged."""
    reset(env)


def task_completion_check(env):
    start_z = getattr(env, "_nav_bottle_start_z", None)
    if start_z is None:
        return False
    bottle = env.scene.object_registry("name", "water_bottle")
    if bottle is None:
        return False
    bottle_pos, _ = bottle.get_position_orientation()
    return (float(bottle_pos[2]) - start_z) >= _LIFT_Z


def register_teleop_keys(env, kb):
    """Teleop post-setup hook: show the robot's head camera in a docked side viewport."""
    import omnigibson.lazy as lazy
    from omnigibson.sensors import VisionSensor
    from omnigibson.utils.ui_utils import dock_window

    try:
        cam = next((s for s in env.robots[0].sensors.values()
                    if isinstance(s, VisionSensor)), None)
        if cam is None:
            print("[door_slam_vase] no robot camera found; skipping head-camera viewport")
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
        print(f"[door_slam_vase] head camera '{cam.name}' shown in '{cam._viewport.name}'")
    except Exception as e:
        print(f"[door_slam_vase] could not show head-camera viewport: {e}")
