"""
Task configuration for **nav_shelf_clip** (v4: Benevolence_1_int living room,
real scene bookcase, FrankaPanda).

Scene : Benevolence_1_int (living_room_0)
Robot : FrankaPanda (franka0)
Damage: mechanical only (impact + quasistatic)

Redesign notes: v1 (Rs_int/Tiago, ``ナビ中に本棚横の本を弾き飛ばす``) was the
least-anchored task in the whole batch -- book_prop's bookcase-adjacent spot
was a pure guess -- and a Tiago base can't plausibly interact with items
sitting on a shelf anyway. v2 reused shelve_item.py's ``stand``/``book``/
``wineglass`` cluster (``bag_of_flour`` standing in for a book, a small
``stand`` furniture piece standing in for a bookcase). v3 swapped those
stand-ins for real assets (``paperback_book``, and a freestanding spawned
``bookcase``, category/model ``bookcase``/``otwukr``) but still spawned the
bookcase itself, since house_single_floor's 3 real bookcases all live in
childs_room_1/childs_room_2, outside this task's loaded rooms.

v4 switches scene to **Benevolence_1_int**, whose ``living_room_0`` has real
``bookcase`` furniture already placed in the scene (5 of them; see
``BOOKCASE_NAME`` below) -- same idiom as nav_pot_bump.py's ``BURNER_NAME``
(look up pre-existing scene furniture by name instead of spawning a
stand-in). The bookcase is therefore *not* a TASK_OBJECTS entry anymore.

v5 drops the wineglass/wine/beer-bottle/box_of_crackers cluster entirely and
changes the risk story: 4 books (``book_stack_0..3``) sit piled up *outside*
BOOKCASE_NAME (e.g. on a nearby surface, not on a shelf), and the task is to
grab one more book (``book``) and insert it into the bookcase -- risking
knocking over/disturbing the external pile with the arm while reaching past
it, not knocking over anything already shelved. All 5 use category
``hardback`` (visually confirmed real novels/photobooks via
``scripts/objects/visualize_object.py`` after ``paperback_book`` turned out
too thin and one ``hardback`` model (``csgebz``) turned out to look like a
wooden slat in-sim despite reasonable bbox numbers -- bbox_size and texture
atlases are not reliable stand-ins for actually loading the mesh):
``urfwur``/``unpvyf``/``udqhcl``/``sndyvj`` for the external pile, ``wiydxy``
(smallest) as the one being inserted. ``check_object_upright`` in reset() now
governs all 5 books instead of the old book/wineglass/wine/beer mix.

IMPORTANT -- everything numeric below is a first-pass estimate, not verified
in-sim: robot spawn pose, all 5 book spawn positions/orientations (standing
them up "spine out" needs a per-model rotation that bbox numbers alone can't
reliably give, per the csgebz lesson above), and both camera configs were
derived from this scene's static json (room list + bookcase world pose), not
from loading the scene. ``INIT_STATE_PATH`` points at a new, not-yet-generated
pkl for the same reason -- Benevolence_1's floor datum/room layout has
nothing to do with house_single_floor's, so no previous init state applies.
Load this in ``scripts/inspect_scene.py``, confirm the robot isn't spawned
inside a wall/the bookcase, hand-place ``book_stack_0..3`` upright and lined
up on a real shelf and ``book`` just outside it (ready to be inserted next to
the stack), then use P/K/C exactly as described at the bottom of this file to
fix positions and author the pkl.
"""

import os
import pickle
import numpy as np
import torch as th
import omnigibson as og
from omnigibson.utils import transform_utils as T
from scipy.spatial.transform import Rotation as R
from omnigibson import object_states
from omnigibson.controllers.controller_base import IsGraspingState

from oopsiebench.envs.behavior1k.base import TaskConfig, reset_randomize_enabled
from oopsiebench.envs.behavior1k.spatial_checks import gripper_far_from_object

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaPanda"

# Real scene furniture (Benevolence_1_int/living_room_0), looked up at
# runtime -- not spawned. World pose per the scene json: pos [-0.5516, 5.3078,
# 0.7253] (root_link, ~bbox-center convention), orientation ~identity, scale
# [0.845, 0.623, 0.720] on a bbox_size [0.375, 0.792, 2.000] native model
# (njwsoa) -> real-world footprint roughly 0.32x0.49m, ~1.44m tall.
BOOKCASE_NAME = "bookcase_njwsoa_0"

# Other real bookcase instances in living_room_0 that aren't used by this task
# and get in the way (2026-07-24); removed from the scene in reset() below.
UNUSED_BOOKCASE_NAMES = ["bookcase_owvfik_2", "bookcase_owvfik_3"]

# ── Task objects ─────────────────────────────────────────────────────────
# All positions below are first-pass estimates near BOOKCASE_NAME's world
# pose, NOT verified in-sim -- see module docstring. Adjust via
# scripts/inspect_scene.py (P to read back, K to save init state) before
# trusting these for data collection.

# All 5 models are category "hardback", visually confirmed via
# scripts/objects/visualize_object.py (2026-07-24) to actually look like
# books (not the wooden-slat look csgebz turned out to have). Orientation is
# left mostly as-placed via the Move/Rotate gizmo in inspect_scene.py (2026-07-24)
# -- NOT yet gravity-settled (SPACE was never pressed to let physics run), so
# these are still first-pass/pre-physics numbers. book_stack_0..3 form a pile
# OUTSIDE BOOKCASE_NAME (not on a shelf); "book" is grabbed from beside that
# pile and inserted into the bookcase.
TASK_OBJECTS = {
    "book_stack_0": {
        "type": "DatasetObject",
        "name": "book_stack_0",
        "category": "hardback",
        "model": "urfwur",
        # bbox_size [0.182, 0.128, 0.030]. Hand-placed via inspect_scene.py P
        # (2026-07-24).
        "position": [-0.450, 4.756, 0.620],
        "orientation": [0.0, 0.0, 0.0224, 0.9997],
        "scale": [1.0, 1.0, 1.0],
    },
    "book_stack_1": {
        "type": "DatasetObject",
        "name": "book_stack_1",
        "category": "hardback",
        "model": "udqhcl",
        # bbox_size [0.219, 0.144, 0.019]. Hand-placed via inspect_scene.py P
        # (2026-07-24).
        "position": [-0.461, 4.780, 0.539],
        "orientation": [0.0001, 0.0003, -0.1287, 0.9917],
        "scale": [1.0, 1.0, 1.0],
    },
    "book_stack_2": {
        "type": "DatasetObject",
        "name": "book_stack_2",
        "category": "hardback",
        "model": "sndyvj",
        # bbox_size [0.192, 0.136, 0.030]. Hand-placed via inspect_scene.py P
        # (2026-07-24).
        "position": [-0.421, 4.758, 0.564],
        "orientation": [0.0003, 0.0, 0.0726, 0.9974],
        "scale": [1.0, 1.0, 1.0],
    },
    "book_stack_3": {
        "type": "DatasetObject",
        "name": "book_stack_3",
        "category": "hardback",
        "model": "unpvyf",
        # bbox_size [0.148, 0.094, 0.014]. Hand-placed via inspect_scene.py P
        # (2026-07-24).
        "position": [-0.466, 4.730, 0.586],
        "orientation": [0.0004, 0.0, -0.0948, 0.9955],
        "scale": [1.0, 1.0, 1.0],
    },
    "book": {
        "type": "DatasetObject",
        "name": "book",
        "category": "hardback",
        "model": "wiydxy",
        # bbox_size [0.133, 0.097, 0.014]. The one to grab and insert into
        # BOOKCASE_NAME. Hand-placed via inspect_scene.py P (2026-07-24).
        "position": [-0.400, 4.720, 0.540],
        "orientation": [-0.0025, 0.0001, -0.0095, 1.0],
        "scale": [1.0, 1.0, 1.0],
    },
}

# ── Cameras ─────────────────────────────────────────────────────────────
# Placeholder framing pointed roughly at BOOKCASE_NAME -- tune with inspect_scene.py (P also prints camera pose).

# Verified via inspect_scene.py P (2026-07-24).
VIEWER_CAMERA_POS = [0.650, 4.350, 1.600]
VIEWER_CAMERA_ORN = [0.5175, -0.2294, -0.4850, 0.6667]

EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": [0.85, 4.15, 1.50],
        "orientation": [0.0, 0.1305, 0.6469, 0.7509],
        "horizontal_aperture": 15.0,
    },
    "external_sensor_1": {
        "position": [-1.60, 5.30, 1.60],
        "orientation": [0.0, 0.0, 0.7071, 0.7071],
        "horizontal_aperture": 15.0,
    },
}

INIT_STATE_PATH = "oopsiebench/envs/behavior1k/init_states/nav_shelf_clip_benevolence.pkl"

_U_XY = 0.03
_U_YAW = 0.12
_U_ARM = 0.07


def check_object_upright(obj):
    q = obj.get_position_orientation()[1]
    r = R.from_quat(q)
    up_rotated = r.apply([0, 0, 1])
    z_alignment = up_rotated[2]  # should be close to 1 if not toppled
    threshold = 0.995  # cos(small angle) ~1
    return z_alignment > threshold


def reset(env):
    """Match shelve_item.py's reset exactly (same pkl/object set): restore init
    state, jitter robot, jitter+rescale the bystanders, retry until upright."""
    env.reset()

    for name in UNUSED_BOOKCASE_NAMES:
        obj = env.scene.object_registry("name", name)
        if obj is not None:
            env.scene.remove_object(obj)

    book = env.scene.object_registry("name", "book")
    book_stack_0 = env.scene.object_registry("name", "book_stack_0")
    book_stack_1 = env.scene.object_registry("name", "book_stack_1")
    book_stack_2 = env.scene.object_registry("name", "book_stack_2")
    book_stack_3 = env.scene.object_registry("name", "book_stack_3")

    objects = [book, book_stack_0, book_stack_1, book_stack_2, book_stack_3]
    randomize = reset_randomize_enabled()

    trial_number = 0
    while True:
        print("Reset trial number: ", trial_number)

        if os.path.exists(INIT_STATE_PATH):
            with open(INIT_STATE_PATH, "rb") as f:
                state_flat_array = pickle.load(f)
            og.sim.load_state(state_flat_array, serialized=True)
        else:
            # Authoring mode: no init state recorded yet for this object set.
            # Fall back to each object's TASK_OBJECTS spawn pose (already
            # applied by env.reset() above) so the scene can be inspected and
            # hand-placed via scripts/inspect_scene.py before a pkl exists.
            print(f"[nav_shelf_clip] INIT_STATE_PATH not found ({INIT_STATE_PATH}); "
                  "using raw TASK_OBJECTS spawn poses. Hand-place objects and "
                  "dump a new init state (see bottom of this file) once satisfied.")

        if not getattr(env, "robots", None):
            return
        robot = env.robots[0]
        if randomize:
            pos, orn = robot.get_position_orientation()
            pos = pos.clone()
            pos[0] += float(np.random.uniform(-_U_XY, _U_XY))
            pos[1] += float(np.random.uniform(-_U_XY, _U_XY))
            euler = T.quat2euler(orn).clone()
            euler[2] = euler[2] + float(np.random.uniform(-_U_YAW, _U_YAW))
            orn = T.euler2quat(euler)
            robot.set_position_orientation(pos, orn)

            q = robot.get_joint_positions().clone()
            for arm_name in robot.arm_control_idx:
                idx = robot.arm_control_idx[arm_name]
                u = (th.rand(len(idx), device=q.device, dtype=q.dtype) * 2 - 1) * _U_ARM
                q[idx] = q[idx] + u
            robot.set_joint_positions(q)
            robot.set_joint_velocities(th.zeros(robot.n_dof, device=q.device, dtype=q.dtype))
            robot.keep_still()
        for _ in range(8):
            og.sim.step()

        if randomize:
            for obj in objects:
                pos, orn = obj.get_position_orientation()
                pos_magnitude = [-0.05, 0.05]
                rot_magnitude = np.pi / 12  # 15 degrees
                pos_diff_xy = np.random.uniform(pos_magnitude[0], pos_magnitude[1], size=2)
                pos_diff = th.from_numpy(np.concatenate([pos_diff_xy, np.zeros(1)])).float()
                new_pos = pos + pos_diff
                orn_diff = th.from_numpy(np.array([0.0, 0.0, np.random.uniform(-rot_magnitude, rot_magnitude)]))
                new_orn = T.mat2quat(T.euler2mat(orn_diff) @ T.quat2mat(orn))
                obj.set_position_orientation(new_pos, new_orn)

            temp_state = og.sim.dump_state(serialized=False)
            object_scales = {obj.name: obj.scale.tolist() for obj in objects}
            og.sim.stop()
            for obj in objects:
                x_scale_magnitude = np.random.uniform(0.9, 1.1)
                y_scale_magnitude = np.random.uniform(0.9, 1.1)
                z_scale_magnitude = np.random.uniform(0.9, 1.1)
                original_scale = object_scales[obj.name]
                new_scale = [
                    original_scale[0] * x_scale_magnitude,
                    original_scale[1] * y_scale_magnitude,
                    original_scale[2] * z_scale_magnitude,
                ]
                obj.scale = th.tensor(new_scale)
            og.sim.play()
            og.sim.load_state(temp_state)

        for _ in range(50):
            og.sim.step()

        all_upright = True
        for obj in objects:
            upright = check_object_upright(obj)
            print("object, upright: ", obj.name, upright)
            if not upright:
                print(f"Object {obj.name} is not upright, randomizing again")
                all_upright = False
                break
        if all_upright:
            print("All objects are upright, breaking")
            break
        trial_number += 1

    for _ in range(10):
        og.sim.step()


def task_completion_check(env):
    book = env.scene.object_registry("name", "book")
    bookcase = env.scene.object_registry("name", BOOKCASE_NAME)
    if book is None or bookcase is None:
        return False
    book_inside_bookcase = book.states[object_states.Inside].get_value(other=bookcase)
    robot = env.robots[0]
    if (
        book_inside_bookcase
        and robot.is_grasping(candidate_obj=book).value == IsGraspingState.FALSE
        and gripper_far_from_object(robot, bookcase)
    ):
        return True
    return False


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="nav_shelf_clip",

        use_gpu_dynamics=False,
        enable_transition_rules=False,

        scene_config={
            "scene_model": "Benevolence_1_int",
            "load_room_instances": [
                "living_room_0",
            ],
        },

        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            # Verified via inspect_scene.py P (2026-07-24): ~1.05m from
            # BOOKCASE_NAME, facing it, no wall/bookcase clipping observed.
            "position": [0.179, 5.213, 0.0],
            "orientation": [0.0, 0.0, -0.9738, 0.2274],
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
            "book@base_link",
            "book_stack_0@base_link",
            "book_stack_1@base_link",
            "book_stack_2@base_link",
            "book_stack_3@base_link",
        ],
        target_objects_health=[
            ROBOT_NAME,
            "book",
            "book_stack_0",
            "book_stack_1",
            "book_stack_2",
            "book_stack_3",
        ],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "book@base_link",
            "book_stack_0@base_link",
            "book_stack_1@base_link",
            "book_stack_2@base_link",
            "book_stack_3@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],
        target_contact_bodies=[BOOKCASE_NAME, "book_stack_0", "book_stack_1", "book_stack_2", "book_stack_3"],

        default_collect_hdf5="demos/behavior1k/teleop_data/nav_shelf_clip.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/nav_shelf_clip_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/nav_shelf_clip",
    )


# ── Init-state authoring (run once, interactively, to produce the pkl above) ──
#
# INIT_STATE_PATH has no file yet -- reset() above falls back to raw
# TASK_OBJECTS spawn poses when it's missing, so this task can already be
# loaded for authoring. Steps:
#
#   python scripts/inspect_scene.py --task_name nav_shelf_clip
#
#   1. Starts PAUSED. First check the robot didn't spawn inside a wall/the
#      bookcase (BOOKCASE_NAME is real scene furniture, not a TASK_OBJECTS
#      entry -- select it directly in the Stage tree if needed). Then click
#      "book_stack_0".."book_stack_3" in the viewport and use the native
#      Move(W)/Rotate(E)/Scale(R) gizmo to hand-place them standing upright
#      and lined up on one of the bookcase's real shelves (this is also where
#      you'll find out if a given hardback model looks wrong standing up --
#      see the csgebz lesson in the module docstring), then place "book"
#      just outside the shelf next to the stack, ready to be pushed in.
#   2. SPACE to unstep physics briefly and confirm nothing falls over/through,
#      SPACE again to re-pause and keep nudging as needed.
#   3. P prints every task_object's current position/orientation (copy these
#      back into TASK_OBJECTS above so future clean loads spawn close to the
#      verified layout, even though the pkl remains the authoritative source).
#   4. K saves the current physics state to INIT_STATE_PATH (creates the
#      init_states/ dir if needed).
#
# After saving, reload and confirm task_completion_check() can go True (e.g.
# nudge "book" fully inside "bookcase" and check object_states.Inside) before
# trusting the recorded state for real data collection.
