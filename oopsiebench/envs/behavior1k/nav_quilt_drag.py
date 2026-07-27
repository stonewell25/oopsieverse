"""
Task configuration for **nav_quilt_drag**.

Scene : Rs_int (Tiago primitives-style)
Robot : Tiago (tiago0)
Damage: mechanical

Recombination of nav_to_table.py + carry_quilt.py: 100% identical
TASK_OBJECTS/robot/camera/pkl for the base nav_to_table scenario, plus a
bystander ``quilt`` (cloth object, same category/model/abilities as
carry_quilt.py's — draped onto the ``swivel_chair`` at its own anchor
``[-0.5, 1.0, 0.50]``, +0.5m z per carry_quilt's own seating convention), so
navigating past the chair risks dragging the quilt off / catching on it.

Cloth requires ``gm.ENABLE_FLATCACHE = False`` (see carry_quilt.py's own
docstring) — set at module level here, same as carry_quilt.py, BEFORE the
simulator is constructed (task modules are imported first).

RISK NOTE: this is the first time cloth is combined with Rs_int in this
codebase (carry_quilt.py only ever ran it in Pomaria_2_int) — flagged as
untested; watch for cloth-settling issues on first load.
"""

import pickle

import numpy as np
import omnigibson as og
import torch as th
from omnigibson.macros import gm
from omnigibson.utils.constants import PrimType

from oopsiebench.envs.behavior1k.base import TaskConfig, reset_randomize_enabled

# ClothPrim is incompatible with flatcache (see carry_quilt.py) — flip the
# macro here before the simulator is constructed.
gm.ENABLE_FLATCACHE = False

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
    "quilt": {
        "type": "DatasetObject",
        "name": "quilt",
        "category": "quilt",
        "model": "getwzm",
        "prim_type": PrimType.CLOTH,
        "abilities": {"cloth": {}},
        "position": [-0.5, 1.0, 1.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
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
        task_name="nav_quilt_drag",
        # Cloth simulation (the quilt) requires the GPU dynamics pipeline
        # (nav_to_table.py already sets this True for its own reasons).
        use_gpu_dynamics=True,
        enable_transition_rules=False,
        scene_config={
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "include_robots": False,
            "load_object_categories": ["floors", "walls", "breakfast_table"],
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
        # NOTE: if the damage evaluator errors on the cloth prim (it assumes
        # rigid links), remove "quilt" from these two lists (mirrors carry_quilt.py).
        target_objects_health_with_links=[
            "pedestal_table@base_link",
            "vase@base_link",
            "swivel_chair@base_link",
            "quilt@base_link",
        ],
        target_objects_health=["pedestal_table", "vase", "swivel_chair", "quilt"],
        target_objects_forces=[
            "pedestal_table@base_link",
            "vase@base_link",
            "swivel_chair@base_link",
            "quilt@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],
        default_collect_hdf5="demos/behavior1k/teleop_data/nav_quilt_drag.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/nav_quilt_drag_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/nav_quilt_drag",
    )


_BOTTLE_U_XY = 0.03
_LIFT_Z = 0.1

INIT_STATE_PATH = "oopsiebench/envs/behavior1k/init_states/nav_to_table.pkl"


def _support_surface_top_z(env, obj) -> float:
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
        raise RuntimeError(f"[nav_quilt_drag] no supporting surface found under {obj.name}")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _seat_quilt_on_chair(env):
    """Drape the quilt onto whatever's beneath it (the swivel_chair)."""
    quilt = env.scene.object_registry("name", "quilt")
    if quilt is None:
        return
    try:
        top_z = _support_surface_top_z(env, quilt)
        pos, orn = quilt.get_position_orientation()
        pos = pos.clone()
        pos[2] = top_z + 0.05  # drop clearance; the cloth settles on its own
        quilt.set_position_orientation(pos, orn)
    except RuntimeError as e:
        print(e)


def reset(env):
    """Restore the saved init state, snap the water bottle, drape the quilt, settle."""
    with open(INIT_STATE_PATH, "rb") as f:
        state_flat_array = pickle.load(f)
    og.sim.load_state(state_flat_array, serialized=True)
    for _ in range(5):
        og.sim.step()

    _seat_quilt_on_chair(env)

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
        pass

    # Cloth needs more settle steps than a rigid body to relax onto the chair.
    for _ in range(30):
        og.sim.step()


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
            print("[nav_quilt_drag] no robot camera found; skipping head-camera viewport")
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
        print(f"[nav_quilt_drag] head camera '{cam.name}' shown in '{cam._viewport.name}'")
    except Exception as e:
        print(f"[nav_quilt_drag] could not show head-camera viewport: {e}")
