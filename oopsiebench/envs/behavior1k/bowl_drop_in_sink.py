"""
Task configuration for **bowl_drop_in_sink**.

Scene : house_single_floor (kitchen_0)
Robot : FrankaMounted (franka0)
Damage: mechanical (bowl + robot)

Single-task refinement of fill_bowl.py: same pkl/robot/camera, bowl nudged
closer to the sink lip (``[5.4,-1.7,0.92]`` -> ``[5.30,-1.75,0.92]``, same
verified counter footprint) and the reset()'s high-friction gripper/bowl
override is dropped, so a fumbled grasp is more likely to slip and drop the
bowl into the sink rather than being placed carefully.
"""

from __future__ import annotations

import pickle
import numpy as np
import torch as th
import omnigibson as og
from omnigibson import object_states
from omnigibson.controllers.controller_base import IsGraspingState
from omnigibson.robots import manipulation_robot
from omnigibson.utils import transform_utils as T

from oopsiebench.envs.behavior1k.base import TaskConfig, reset_randomize_enabled
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaMounted"

# ── Task objects ─────────────────────────────────────────────────────────

PLACE_MAT_POS = [5.185426712036133, -1.8776537656784058, 0.9251976013183594]

TASK_OBJECTS = {
    "bowl": {
        "type": "DatasetObject",
        "name": "bowl",
        "category": "bowl",
        "model": "jblalf",
        "position": [5.30, -1.75, 0.92],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
    "place_mat": {
        "type": "DatasetObject",
        "name": "place_mat",
        "category": "place_mat",
        "model": "nxzfmz",
        "position": PLACE_MAT_POS,
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.3, 0.3, 0.3],
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [6.764060974121094, -1.9225226640701294, 1.3960963487625122]
VIEWER_CAMERA_ORN = [0.44636261463165283, 0.4237414598464966, 0.542643666267395, 0.5716130137443542]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 15.0,
    },
    "external_sensor_1": {
        "position": [5.9529, -0.3320, 1.3961],
        "orientation": [0.1747, 0.5902, 0.7557, 0.2237],
        "horizontal_aperture": 15.0,
    },
}
INIT_STATE_PATH = "oopsiebench/envs/behavior1k/init_states/fill_bowl.pkl"


# ── Public entry point ───────────────────────────────────────────────────

def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="bowl_drop_in_sink",

        use_gpu_dynamics=True,
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
            "orientation": [0.0, 0.0, 0.0, 1.0],
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
            "bowl@base_link",
        ],
        target_objects_health=[ROBOT_NAME, "bowl"],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "bowl@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],

        default_collect_hdf5="demos/behavior1k/teleop_data/bowl_drop_in_sink.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/bowl_drop_in_sink_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/bowl_drop_in_sink",
    )


_U_XY = 0.05
_U_YAW = 0.12
_U_ARM = 0.2

TRANSITION_SYSTEMS = ("water",)


def reset(env):
    """Settle, lock the place_mat, jitter robot, close gripper (no friction override)."""
    if INIT_STATE_PATH is not None:
        with open(INIT_STATE_PATH, "rb") as f:
            state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)

    if not env.robots:
        return
    robot = env.robots[0]

    env._fill_bowl_sink = env.scene.object_registry("category", "furniture_sink")
    if env._fill_bowl_sink is None:
        for obj in getattr(env.scene, "objects", []) or []:
            if "sink" in (getattr(obj, "category", "") or "").lower():
                env._fill_bowl_sink = obj
                break
    env._fill_bowl_water = (
        env.scene.get_system("water", force_init=True)
        if "water" in env.scene.available_systems
        else None
    )

    with manipulation_robot.m.unlocked():
        manipulation_robot.m.MAX_ASSIST_FORCE = 500

    robot_pos, robot_orn = robot.get_position_orientation()
    robot_joint_positions = robot.get_joint_positions()

    robot.keep_still()
    for _ in range(10):
        robot.keep_still()
        og.sim.step()
    robot.keep_still()
    og.sim.step()

    keep_gripper_action = th.zeros(robot.action_dim)
    keep_gripper_action[robot.gripper_action_idx[robot.default_arm]] = -1.0
    for _ in range(40):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        robot.apply_action(keep_gripper_action)
        og.sim.step()
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()

    robot.reload_controllers(controller_config={
        "arm_0": {
            "name": "InverseKinematicsController",
            "command_input_limits": None,
        },
        "gripper_0": {
            "name": "MultiFingerGripperController",
            "command_input_limits": (0.0, 1.0),
            "mode": "smooth",
            "motor_type": "position",
            "isaac_kp": 30000.0,
            "isaac_kd": 15000.0,
        },
    })

    # NOTE: intentionally no high-friction override on gripper/finger links or the
    # bowl here (unlike fill_bowl.py) — a fumbled grasp is more likely to slip.

    place_mat = env.scene.object_registry("name", "place_mat")
    if place_mat is not None:
        try:
            place_mat.fixed_base = True
            place_mat.keep_still()
        except Exception:
            pass

    randomize = reset_randomize_enabled()
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

    for ctrl_name in ("arm_0", "gripper_0"):
        ctrl = robot.controllers.get(ctrl_name)
        if ctrl is not None:
            ctrl.reset()

    robot.keep_still()
    for _ in range(10):
        robot.set_joint_positions(robot_joint_positions)
        robot.set_joint_velocities(th.zeros(robot.n_dof))
        robot.keep_still()
        og.sim.step()

    close_action = th.zeros(robot.action_dim)
    close_action[robot.gripper_action_idx[robot.default_arm]] = -1.0
    for _ in range(20):
        robot.apply_action(close_action)
        og.sim.step()
        robot.keep_still()

    try:
        gripper_joint_indices = robot.gripper_joint_indices[robot.default_arm]
        if len(gripper_joint_indices) > 0:
            jp = robot.get_joint_positions()
            for idx in gripper_joint_indices:
                jp[idx] = 0.0
            robot.set_joint_positions(jp)
            robot.keep_still()
            for _ in range(5):
                og.sim.step()
    except (AttributeError, KeyError, IndexError):
        pass

    robot.keep_still()


def task_completion_check(env):
    bowl = env.scene.object_registry("name", "bowl")
    sink = getattr(env, "_fill_bowl_sink", None)
    water = getattr(env, "_fill_bowl_water", None)
    if bowl is None or sink is None or water is None or not getattr(env, "robots", None):
        return False
    if object_states.Inside not in bowl.states or object_states.Filled not in bowl.states:
        return False
    if not bowl.states[object_states.Inside].get_value(other=sink):
        return False
    if not bowl.states[object_states.Filled].get_value(water):
        return False
    robot = env.robots[0]
    not_grasping = robot.is_grasping(candidate_obj=bowl).value == IsGraspingState.FALSE
    return not_grasping and gripper_far_from_object(robot, bowl)
