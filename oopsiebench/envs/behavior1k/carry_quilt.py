"""
Task configuration for **carry_quilt**.

Scene : Pomaria_2_int (bedroom_0's own bed, ``bed_mpgavt_0``, as the support surface)
Robot : FrankaMounted (franka0)
Damage: mechanical (quilt + robot)

The quilt is injected as a **cloth** object: ``prim_type=PrimType.CLOTH`` +
``abilities={"cloth": {}}``, mirroring OmniGibson's own
``examples/object_states/folded_unfolded_state_demo.py``. Without those two
keys a DatasetObject loads as a plain rigid body (first version of this task
made exactly that mistake — the quilt was rock hard). Cloth requires
``use_gpu_dynamics=True``.

CAVEATS (untested):
- Assisted grasping on cloth may not latch like on rigid bodies; if it
  doesn't, try ``grasping_mode: "sticky"`` or OG's cloth attachment APIs.
- The damage evaluator path (DamageableDatasetObject / qs-force tracking)
  assumes rigid links; if it errors on the cloth prim, drop "quilt" from the
  health/forces target lists below and keep it decorative.

No init_states pkl is authored yet (``INIT_STATE_PATH = None``) — reset()
seats the quilt above the bed and lets the cloth settle. The robot spawn
position came from interactive tuning in scripts/inspect_scene.py.
"""

from __future__ import annotations

import numpy as np
import omnigibson as og
import torch as th
from omnigibson.macros import gm
from omnigibson.utils import transform_utils as T
from omnigibson.utils.constants import PrimType

from oopsiebench.envs.behavior1k.base import TaskConfig

# ClothPrim is incompatible with flatcache (cloth_prim.py:103 asserts
# ``not gm.ENABLE_FLATCACHE``), and OG's default is True. TaskConfig has no
# field for this, but task modules are imported (load_task_config) BEFORE the
# env/simulator is constructed, so flipping the macro here takes effect for
# teleop_b1k / playback_b1k / inspect_scene alike without touching them.
gm.ENABLE_FLATCACHE = False

ROBOT_NAME = "franka0"
ROBOT_TYPE = "FrankaMounted"

INIT_STATE_PATH = None

# ── Task objects ─────────────────────────────────────────────────────────

# bed_mpgavt_0 (native to Pomaria_2_int, bedroom_0, fixed_base) sits at
# world xy ~= (-0.34, -1.90). The quilt spawns above it; reset() seats it
# onto the bed's top surface via runtime AABB lookup and lets it drape.
_BED_XY = (-0.3370, -1.9013)

TASK_OBJECTS = {
    "quilt": {
        "type": "DatasetObject",
        "name": "quilt",
        "category": "quilt",
        "model": "getwzm",
        "prim_type": PrimType.CLOTH,
        "abilities": {"cloth": {}},
        "position": [_BED_XY[0]+0.5, _BED_XY[1], 1.5],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    },
}

# ── Cameras ──────────────────────────────────────────────────────────────

VIEWER_CAMERA_POS = [2.216160774230957, -1.7015305757522583, 1.880691647529602]
VIEWER_CAMERA_ORN = [0.3691592812538147, 0.4569842219352722, 0.6030932068824768, 0.5395974516868591]


EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor_0": {
        "position": VIEWER_CAMERA_POS,
        "orientation": VIEWER_CAMERA_ORN,
        "horizontal_aperture": 15.0,
    },
    "external_sensor_1": {
        "position": [_BED_XY[0] + 1.3, _BED_XY[1] + 1.0, 1.6],
        "orientation": [0.32, -0.32, -0.63, -0.63],
        "horizontal_aperture": 15.0,
    },
}

_U_XY = 0.03
_U_YAW = 0.12
_U_ARM = 0.07
_LIFT_Z = 0.1


def _support_surface_top_z(env, obj) -> float:
    """Top-z of the nearest surface directly beneath obj's xy (e.g. the bed)."""
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
        raise RuntimeError(f"[carry_quilt] no supporting surface found under {obj.name}")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _seat_on_bed(env, obj):
    obj_pos, obj_orn = obj.get_position_orientation()
    top_z = _support_surface_top_z(env, obj)
    pos = obj_pos.clone()
    pos[2] = top_z + 0.05  # a bit of drop clearance; the cloth settles on its own
    obj.set_position_orientation(pos, obj_orn)


def reset(env):
    """Seat the quilt on the bed; jitter the Franka base/arm; settle (cloth needs extra steps)."""
    if not getattr(env, "robots", None):
        return
    robot = env.robots[0]

    for _ in range(5):
        og.sim.step()

    quilt = env.scene.object_registry("name", "quilt")
    if quilt is not None:
        try:
            _seat_on_bed(env, quilt)
        except RuntimeError as e:
            print(e)

    pos, orn = robot.get_position_orientation()
    pos = pos.clone()
    pos[0] += float(np.random.uniform(-_U_XY, _U_XY))
    pos[1] += float(np.random.uniform(-_U_XY, _U_XY))
    euler = T.quat2euler(orn).clone()
    euler[2] = euler[2] + float(np.random.uniform(-_U_YAW, _U_YAW))
    robot.set_position_orientation(pos, T.euler2quat(euler))

    q = robot.get_joint_positions().clone()
    for arm_name in robot.arm_control_idx:
        idx = robot.arm_control_idx[arm_name]
        u = (th.rand(len(idx), device=q.device, dtype=q.dtype) * 2 - 1) * _U_ARM
        q[idx] = q[idx] + u
    robot.set_joint_positions(q)
    robot.set_joint_velocities(th.zeros(robot.n_dof, device=q.device, dtype=q.dtype))
    robot.keep_still()

    # Cloth needs more settle steps than a rigid body to relax onto the bed.
    for _ in range(30):
        og.sim.step()

    quilt = env.scene.object_registry("name", "quilt")
    if quilt is not None:
        p, _ = quilt.get_position_orientation()
        env._carry_quilt_start_z = float(p[2])
    else:
        env._carry_quilt_start_z = None


def playback_reset(env):
    """reset() isn't run during playback — re-seat the quilt so it starts on the bed."""
    quilt = env.scene.object_registry("name", "quilt")
    if quilt is None:
        return
    try:
        _seat_on_bed(env, quilt)
    except RuntimeError as e:
        print(e)


def task_completion_check(env):
    """Best-effort: quilt centroid lifted off the bed by _LIFT_Z.

    Not gated on ``is_grasping`` — AG's grasp-state detection is tuned for
    rigid bodies and may not reliably report True for a cloth object even
    while the gripper is genuinely holding it pinched.
    """
    start_z = getattr(env, "_carry_quilt_start_z", None)
    if start_z is None:
        return False
    quilt = env.scene.object_registry("name", "quilt")
    if quilt is None:
        return False
    quilt_pos, _ = quilt.get_position_orientation()
    return (float(quilt_pos[2]) - start_z) >= _LIFT_Z


def register_teleop_keys(env, kb):
    """Teleop post-setup hook: show the robot's onboard (EEF) camera in a docked side viewport."""
    import omnigibson.lazy as lazy
    from omnigibson.sensors import VisionSensor
    from omnigibson.utils.ui_utils import dock_window

    try:
        cam = next((s for s in env.robots[0].sensors.values()
                    if isinstance(s, VisionSensor)), None)
        if cam is None:
            print("[carry_quilt] no robot camera found; skipping onboard-camera viewport")
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
        print(f"[carry_quilt] robot camera '{cam.name}' shown in '{cam._viewport.name}'")
    except Exception as e:
        print(f"[carry_quilt] could not show onboard-camera viewport: {e}")


def get_task_config() -> TaskConfig:
    return TaskConfig(
        task_name="carry_quilt",

        # Cloth simulation (the quilt) requires the GPU dynamics pipeline.
        use_gpu_dynamics=True,
        enable_transition_rules=False,

        scene_config={
            "scene_model": "Pomaria_2_int",
            "load_object_categories": ["floors", "walls", "ceilings", "bed"],
            "load_room_instances": ["bedroom_0"],
        },

        robot_name=ROBOT_NAME,
        robot_type=ROBOT_TYPE,
        robot_config={
            "type": ROBOT_TYPE,
            "name": ROBOT_NAME,
            "position": [0.6629999876022339, -1.9012999534606934, -1.4901161193847656e-07],
            "orientation": [0.0, 0.0, 1.0000001192092896, 6.123234924670329e-17],
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

        # NOTE: if the damage evaluator errors on the cloth prim (it assumes
        # rigid links), remove "quilt" from these three lists.
        target_objects_health_with_links=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "quilt@base_link",
        ],
        target_objects_health=[ROBOT_NAME, "quilt"],
        target_objects_forces=[
            f"{ROBOT_NAME}@eef_link",
            f"{ROBOT_NAME}@panda_hand",
            f"{ROBOT_NAME}@panda_leftfinger",
            f"{ROBOT_NAME}@panda_rightfinger",
            "quilt@base_link",
        ],
        force_keys=["filtered_qs_forces", "impact_forces"],

        default_collect_hdf5="demos/behavior1k/teleop_data/carry_quilt.hdf5",
        default_playback_hdf5="demos/behavior1k/playback_data/carry_quilt_playback.hdf5",
        default_video_dir="demos/behavior1k/playback_videos/carry_quilt",
    )
