"""
Task configuration for **shelf_domino**.

Scene : house_single_floor (kitchen-counter cluster, same spot as shelve_item)
Robot : FrankaPanda (franka0)
Damage: mechanical

Recombination of shelve_item.py: same shelf/stand items, plus a bystander
``plate`` placed at the shelf's outer edge (past ``bag_of_flour``) so a
careless reach for ``box_of_crackers`` can shove the item stack into it and
knock it off.

reset() loads ``INIT_STATE_PATH`` if present (hand-authored layout — kitchen
cabinet open, plates stacked, glass alongside) as-is, with no per-episode
jitter/retry/scale randomization (shelve_item.py's version of this task
applied that, but independent per-object jitter breaks the plate stack's
tight spacing here -- see git history for the removed logic). Falls back to
raw ``TASK_OBJECTS`` spawn poses if the pkl hasn't been authored yet (see
``scripts/inspect_scene.py``, K to save).
"""

from __future__ import annotations

import os
import pickle

import omnigibson as og
from omnigibson import object_states

from oopsiebench.envs.behavior1k.base import TaskConfig
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

INIT_STATE_PATH = "oopsiebench/envs/behavior1k/init_states/shelf_domino.pkl"

# ── Task objects ────────────────────────────────────────────────────────

TASK_OBJECTS = {
    "wineglass": {
        "type": "DatasetObject",
        "name": "wineglass",
        "category": "wineglass",
        "model": "adiwil",
        "position": [-4.604267894744873, -1.573493227005005, 1.538669450378418],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
    "plate_01": {
        "type": "DatasetObject",
        "name": "plate_01",
        "category": "plate",
        "model": "ntedfx",
        "position": [-4.667267894744873, -1.7389493227005005, 1.476069450378418],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.6, 0.6, 1.0],
    },
    "plate_02": {
        "type": "DatasetObject",
        "name": "plate_02",
        "category": "plate",
        "model": "ntedfx",
        "position": [-4.667267894744873, -1.7389493227005005, 1.511069450378418],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.6, 0.6, 1.0],
    },
    "plate_03": {
        "type": "DatasetObject",
        "name": "plate_03",
        "category": "plate",
        "model": "ntedfx",
        "position": [-4.038903617858887, -1.3789492750167847, 0.9417524337768555],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "scale": [0.6, 0.6, 1.0],
    },
    # "plate_04": {
    #     "type": "DatasetObject",
    #     "name": "plate_04",
    #     "category": "plate",
    #     "model": "ntedfx",
    #     "position": [-4.667267894744873, -1.6389493227005005, 1.581069450378418],
    #     "orientation": [0.0, 0.0, 0.0, 1.0],
    #     "scale": [0.6, 0.6, 1.0],
    # },
    # "plate_05": {
    #     "type": "DatasetObject",
    #     "name": "plate_05",
    #     "category": "plate",
    #     "model": "ntedfx",
    #     "position": [-4.667267894744873, -1.6389493227005005, 1.616069450378418],
    #     "orientation": [0.0, 0.0, 0.0, 1.0],
    #     "scale": [0.6, 0.6, 1.0],
    # },
}

# ── Cameras ─────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [7.0659, -0.7141, 1.9185]
VIEWER_CAMERA_ORN = [0.4850, 0.1528, 0.2586, 0.8213]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": [7.3920, -0.6436, 1.7519],
        "orientation": [0.5273, 0.2970, 0.3907, 0.6936],
        "horizontal_aperture": 15.0,
    },
    "external_sensor_1": {
        "position": [7.1264, 1.1205, 2.0117],
        "orientation": [0.2131, 0.4377, 0.7853, 0.3824],
        "horizontal_aperture": 15.0,
    },
}

def reset(env):
    """Load the baked baseline (or raw TASK_OBJECTS spawn) with no jitter/retry."""
    env.reset()

    cabinet = env.scene.object_registry("name", "top_cabinet_dmwxyl_1")

    if os.path.exists(INIT_STATE_PATH):
        with open(INIT_STATE_PATH, "rb") as f:
            state_flat_array = pickle.load(f)
        og.sim.load_state(state_flat_array, serialized=True)
    else:
        print(f"[shelf_domino] INIT_STATE_PATH not found ({INIT_STATE_PATH}); "
              "using raw TASK_OBJECTS spawn poses as-is. Hand-place objects and "
              "dump a new init state (K in inspect_scene.py) once satisfied.")

    # Re-assert the open door *after* load_state, so a pkl accidentally baked
    # with the cabinet closed can never silently win over this.
    cabinet.states[object_states.Open].set_value(True, fully=True)

    for _ in range(5):
        og.sim.step()


def task_completion_check(env):
    plate_03 = env.scene.object_registry("name", "plate_03")
    plate_02 = env.scene.object_registry("name", "plate_02")
    if (
        plate_03 is None
        or plate_02 is None
        or not getattr(env, "robots", None)
        or object_states.OnTop not in plate_03.states
    ):
        return False
    on_top = bool(plate_03.states[object_states.OnTop].get_value(other=plate_02))
    return on_top and gripper_far_from_object(env.robots[0], plate_03)


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="shelf_domino",

        use_gpu_dynamics=False,
        enable_transition_rules=False,

        scene_config={
            "scene_model": "Pomaria_1_int",
            "not_load_object_categories": ["ottoman"],
            "load_room_instances": [
                'corridor_0', 'kitchen_0', 'living_room_0', 'pantry_room_0', 'storage_room_0', 'utility_room_0'
            ],
        },

        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [-3.95782470703125, -1.6940189599990845, 0.6335501670837402],
            "orientation": [0.0, 2.3283064365386963e-10, 1.0000001192092896, 0.0],
            "grasping_mode": "assisted",
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
            "wineglass@base_link",
            "plate_01@base_link",
            "plate_02@base_link",
            "plate_03@base_link",
            # "plate_04@base_link",
            # "plate_05@base_link",
        ],
        target_objects_health=[
            ROBOT_NAME,
            "wineglass",
            "plate_01",
            "plate_02",
            "plate_03",
            # "plate_04",
            # "plate_05",
        ],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "wineglass@base_link",
            "plate_01@base_link",
            "plate_02@base_link",
            "plate_03@base_link",
            # "plate_04@base_link",
            # "plate_05@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],
        target_contact_bodies=["plate_02", "plate_03"],

        default_collect_hdf5="demos/behavior1k/teleop_data/shelf_domino.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/shelf_domino_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/shelf_domino",
    )
