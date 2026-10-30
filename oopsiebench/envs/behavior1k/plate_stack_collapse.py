"""
Task configuration for **plate_stack_collapse**.

Scene : house_single_floor (kitchen_0)
Robot : FrankaMounted (franka0)
Damage: mechanical (3 stacked plates + robot)

Recombination of place_plate.py: same place_mat/robot spawn/camera/pkl, but
``plate`` is stacked three deep (``plate_bottom`` at place_plate's original
pose, ``plate_middle``/``plate_top`` 5cm/10cm above it) so a careless
approach can topple the stack instead of placing a single plate cleanly. The
completion check retargets to ``plate_bottom`` (place_plate's original logic,
unmodified otherwise).
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson import object_states
from omnigibson.controllers.controller_base import IsGraspingState
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaMounted"

INIT_STATE_PATH = "oopsiebench/envs/behavior1k/init_states/plate_stack_collapse.pkl"

# ── Task objects ─────────────────────────────────────────────────────────

PLACE_MAT_POS = [5.185426712036133, -1.8776537656784058, 0.9251976013183594]

TASK_OBJECTS = {
    "plate_bottom": {
        "type": "DatasetObject",
        "name": "plate_bottom",
        "category": "plate",
        "model": "ntedfx",
        "position": [5.4, -1.7, 0.95],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "plate_middle": {
        "type": "DatasetObject",
        "name": "plate_middle",
        "category": "plate",
        "model": "ntedfx",
        # 0.035m = this model's real bbox thickness (see shelf_domino.py's
        # plate_01/plate_02 spacing, [[oopsieverse-task-authoring]]'s bbox
        # pitfall) -- the original 0.05m gap left plates interpenetrating at
        # spawn, which blew the whole stack off the counter on step 1.
        "position": [5.4, -1.7, 0.985],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "plate_top": {
        "type": "DatasetObject",
        "name": "plate_top",
        "category": "plate",
        "model": "ntedfx",
        "position": [5.4, -1.7, 1.02],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "place_mat": {
        "type": "DatasetObject",
        "name": "place_mat",
        "category": "place_mat",
        "model": "nxzfmz",
        "position": PLACE_MAT_POS,
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.3, 0.3, 0.4],
        "fixed_base": True,
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────


VIEWER_CAMERA_POS = [5.370478630065918, -0.8758299350738525, 1.526785254478454]
VIEWER_CAMERA_ORN = [-0.2920994758605957, 0.48734050989151, 0.7058322429656982, -0.42305791378021244]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 15.0,
    },
    "external_sensor_1": {
        "position": [8.088642120361328, -0.7926759719848633, 1.5118576288223267],
        "orientation": [0.319642573595047, 0.540861189365387, 0.6697852611541748, 0.39583513140678406],
        "horizontal_aperture": 15.0,
    },
}

# ── Public entry point ───────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="plate_stack_collapse",

        use_gpu_dynamics=False,
        enable_transition_rules=False,

        scene_config={
            "scene_model": "house_single_floor",
            "not_load_object_categories": ["ottoman"],
            "load_room_instances": ["kitchen_0"],
        },

        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [5.7, -1.4, 0.0],
            "orientation": [0.0, 0.0, -0.7071067811865476, 0.7071067811865476],
            "grasping_mode": "assisted",
            "obs_modalities": ["rgb", "depth"],
            "action_normalize": False,
            "self_collisions": True,
            "controller_config": {
                "arm_0": {
                    "name": "InverseKinematicsController",
                    "command_input_limits": None,
                },
                "gripper_0": {
                    "name": "MultiFingerGripperController",
                    "command_input_limits": (0.0, 1.0),
                    "mode": "smooth",
                },
            },
        },

        task_objects=TASK_OBJECTS,

        viewer_camera_pos=VIEWER_CAMERA_POS,
        viewer_camera_orn=VIEWER_CAMERA_ORN,
        external_camera_configs=EXTERNAL_CAMERA_CONFIGS,

        target_objects_health_with_links=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "plate_bottom@base_link",
            "plate_middle@base_link",
            "plate_top@base_link",
        ],
        target_objects_health=[ROBOT_NAME, "plate_bottom", "plate_middle", "plate_top"],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "plate_bottom@base_link",
            "plate_middle@base_link",
            "plate_top@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],

        default_collect_hdf5="demos/behavior1k/teleop_data/plate_stack_collapse.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/plate_stack_collapse_playbackhdf5",
        default_video_dir="demos/behavior1k/playback_videos/plate_stack_collapse",
    )


_U_XY = 0.05
_U_YAW = 0.12
_U_ARM = 0.2
_MAT_U_XY = 0.05
_MAT_U_YAW = 0.08
_MAX_RESET_TRIES = 20
# place_plate.py's 0.15m assumed the plate spawns off the mat and must clear a
# height gap before being placed on it. Here the pkl already places
# plate_bottom directly on the counter next to (not on top of) place_mat, so
# a small *negative* clearance (plate resting slightly below the mat's own
# top surface) is the correct, stable resting state, not a failure -- the
# measured value is -0.011m with jitter off (deterministic). -0.05m leaves
# margin for that while still catching a genuine fall-through (old bug: -0.9m).
_MIN_PLATE_CLEARANCE = -0.05

_LOW_BOUNCE_PHYSICS = {
    "static_friction": 0.8,
    "dynamic_friction": 0.7,
    "restitution": 0.0,
}

_PLATE_NAMES = ("plate_bottom", "plate_middle", "plate_top")


def _apply_low_bounce_material(obj, mat_name, **material_kwargs):
    mat_prim_path = f"{obj.prim_path}/Looks/{mat_name}"
    physics_mat = lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
        prim_path=mat_prim_path, name=mat_name, **material_kwargs,
    )
    try:
        stage = lazy.isaacsim.core.utils.stage.get_current_stage()
        mat_prim = stage.GetPrimAtPath(mat_prim_path)
        physx_api = lazy.pxr.PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
        physx_api.CreateRestitutionCombineModeAttr().Set("min")
        physx_api.CreateFrictionCombineModeAttr().Set("max")
    except Exception as e:
        print(f"[plate_stack_collapse] could not set restitution combine mode: {e}")
    for link in obj.links.values():
        for msh in link.collision_meshes.values():
            msh.apply_physics_material(physics_mat)


def _counter_top_z_under(env, xy_pos):
    excluded = {"place_mat", *_PLATE_NAMES}
    for r in getattr(env, "robots", []) or []:
        if hasattr(r, "name"):
            excluded.add(r.name)
    target_z = float(xy_pos[2])
    best = None
    for obj in getattr(env.scene, "objects", []) or []:
        if obj is None or getattr(obj, "name", None) in excluded:
            continue
        if not hasattr(obj, "aabb") or obj.aabb is None:
            continue
        amin, amax = obj.aabb
        top_z = float(amax[2])
        if top_z > target_z + 0.05:
            continue
        if not (float(amin[0]) <= float(xy_pos[0]) <= float(amax[0])
                and float(amin[1]) <= float(xy_pos[1]) <= float(amax[1])):
            continue
        if best is None or top_z > best:
            best = top_z
    return best


def reset(env):
    """Settle robot, snap kinematic mat onto the counter with jitter, jitter robot pose."""
    if not env.robots:
        return
    robot = env.robots[0]

    if not getattr(env, "_low_bounce_applied", False):
        for _name in (*_PLATE_NAMES, "place_mat"):
            _obj = env.scene.object_registry("name", _name)
            if _obj is not None:
                _apply_low_bounce_material(_obj, f"{_name}_low_bounce_mat", **_LOW_BOUNCE_PHYSICS)
        env._low_bounce_applied = True

    # Unlike place_plate (a single plate), this task stacks 3 plates with no
    # gap -- place_plate's inherited jitter (+-0.2 rad per arm joint, +-0.05m
    # place_mat xy) is large enough to knock the arm/mat into the stack and
    # topple it even from a good pkl. shelf_domino hit the same problem and
    # fixed it by removing jitter outright for its stacked-plate scene (see
    # docs/task_authoring_workflow.md); do the same here rather than relying
    # on the global reset_randomize_enabled() toggle.
    randomize = False

    for attempt in range(_MAX_RESET_TRIES):
        if os.path.exists(INIT_STATE_PATH):
            with open(INIT_STATE_PATH, "rb") as f:
                state_flat_array = pickle.load(f)
            og.sim.load_state(state_flat_array, serialized=True)
        else:
            print(f"[plate_stack_collapse] INIT_STATE_PATH not found ({INIT_STATE_PATH}); "
                  "using raw TASK_OBJECTS spawn poses. Hand-place objects (and form a "
                  "grasp via inspect_scene.py's G/N keys) then dump a new init state "
                  "(K in inspect_scene.py) once satisfied.")
        for _ in range(5):
            og.sim.step()

        robot.keep_still()
        for _ in range(5):
            og.sim.step()

        robot_pos, robot_orn = robot.get_position_orientation()
        robot_joint_positions = robot.get_joint_positions()

        # The pkl's arm pose is whatever the sim happened to look like when
        # the layout was captured in inspect_scene.py (often a too-straight,
        # un-posed configuration), not a deliberately chosen "ready" pose.
        # Override just the arm joints with the robot class's own canonical
        # reset_joint_pos (see omnigibson/robots/franka.py's
        # _default_joint_pos) every reset -- plate/mat/robot-base poses from
        # the pkl are left untouched. If a different arm pose is wanted,
        # this is the place to change it (either edit this override, or pass
        # robot_config["reset_joint_pos"]=<array> in get_task_config() and
        # reference that here instead of the class default).
        default_arm_pose = getattr(robot, "reset_joint_pos", None)
        if default_arm_pose is not None:
            for arm_name, idx in robot.arm_control_idx.items():
                robot_joint_positions[idx] = default_arm_pose[idx]

        for ctrl_name in ("arm_0", "gripper_0"):
            ctrl = robot.controllers.get(ctrl_name)
            if ctrl is not None:
                ctrl.reset()

        try:
            is_grasping = robot.is_grasping().value == IsGraspingState.TRUE
        except Exception:
            is_grasping = True
        if not is_grasping:
            try:
                close_action = th.zeros(robot.action_dim)
                close_action[robot.gripper_action_idx[robot.default_arm]] = -1.0
                robot.apply_action(close_action)
            except Exception:
                pass
            robot.keep_still()
            for _ in range(10):
                og.sim.step()

        place_mat = env.scene.object_registry("name", "place_mat")
        if place_mat is not None:
            mpos = th.tensor(PLACE_MAT_POS, dtype=th.float32).clone()
            if randomize:
                mpos[0] += float(np.random.uniform(-_MAT_U_XY, _MAT_U_XY))
                mpos[1] += float(np.random.uniform(-_MAT_U_XY, _MAT_U_XY))
            counter_top_z = _counter_top_z_under(env, mpos)
            if counter_top_z is not None:
                mat_aabb = getattr(place_mat, "aabb", None)
                half_h = 0.5 * float(mat_aabb[1][2] - mat_aabb[0][2]) if mat_aabb is not None else 0.0
                mpos[2] = counter_top_z + half_h + 0.001
            yaw = float(np.random.uniform(-_MAT_U_YAW, _MAT_U_YAW)) if randomize else 0.0
            morn = T.euler2quat(th.tensor([0.0, 0.0, yaw], dtype=th.float32))
            place_mat.set_position_orientation(mpos, morn)
            place_mat.keep_still()

        pos = robot_pos.clone()
        euler = T.quat2euler(robot_orn).clone()
        q = robot_joint_positions.clone()
        if randomize:
            pos[0] += float(np.random.uniform(-_U_XY, _U_XY))
            pos[1] += float(np.random.uniform(-_U_XY, _U_XY))
            euler[2] = euler[2] + float(np.random.uniform(-_U_YAW, _U_YAW))
            for arm_name in robot.arm_control_idx:
                idx = robot.arm_control_idx[arm_name]
                u = (th.rand(len(idx), device=q.device, dtype=q.dtype) * 2 - 1) * _U_ARM
                q[idx] = q[idx] + u
        robot.set_position_orientation(pos, T.euler2quat(euler))
        robot.set_joint_positions(q)
        robot.set_joint_velocities(th.zeros(robot.n_dof))

        robot.keep_still()
        for _ in range(5):
            og.sim.step()

        plate_bottom = env.scene.object_registry("name", "plate_bottom")
        clearance = None
        if (plate_bottom is not None and place_mat is not None
                and getattr(plate_bottom, "aabb", None) is not None
                and getattr(place_mat, "aabb", None) is not None):
            clearance = float(plate_bottom.aabb[0][2]) - float(place_mat.aabb[1][2])

        clearance_str = "n/a" if clearance is None else f"{clearance:.3f}"
        print(f"[plate_stack_collapse] reset attempt {attempt + 1}/{_MAX_RESET_TRIES}: plate-mat clearance = {clearance_str}")
        if clearance is None or clearance >= _MIN_PLATE_CLEARANCE:
            break
    else:
        print(f"[plate_stack_collapse] WARNING: plate clearance below {_MIN_PLATE_CLEARANCE} after {_MAX_RESET_TRIES} attempts")


def task_completion_check(env):
    plate_bottom = env.scene.object_registry("name", "plate_bottom")
    place_mat = env.scene.object_registry("name", "place_mat")
    if (
        plate_bottom is None
        or place_mat is None
        or not getattr(env, "robots", None)
        or object_states.OnTop not in plate_bottom.states
    ):
        return False
    on_top = bool(plate_bottom.states[object_states.OnTop].get_value(other=place_mat))
    return on_top and gripper_far_from_object(env.robots[0], plate_bottom)
