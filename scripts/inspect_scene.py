#!/usr/bin/env python3
"""
Interactive scene inspector: load a task's scene/robot/objects with the
default Isaac-Sim editor layout intact (Stage tree, Property panel, Timeline
all visible/dockable as usual), so robot/camera/object positions can be
eyeballed and tuned by hand before being copied into a task config file.

Deliberately kept separate from scripts/teleop_b1k.py: reuses its
TASK_REGISTRY / config-building helpers via import, but never calls
damagesim.utils.misc_utils.setup_viewport_layout() (which hides every
non-Viewport Kit window) and does no HDF5 collection or robot teleop control.
teleop_b1k.py itself is not modified by this script.

Usage:
    python scripts/inspect_scene.py --task_name nav_to_table_02

Camera/gizmo navigation is deliberately left 100% native Kit -- this script
adds no WASD/camera keybindings of its own (an earlier version used
OmniGibson's CameraMover for that, but CameraMover binds W/A/S/D/T/G
unconditionally, which stole Kit's native W=Move-gizmo / E=Rotate-gizmo
tool hotkeys and native right-click-held+WASD fly-through). With nothing
shadowing them, the usual Isaac Sim controls just work:
    Right-click + drag        -- look around / fly (hold + WASD to move)
    Alt + LMB/MMB/RMB drag     -- orbit / pan / zoom around a pivot
    W / E / R (no right-click) -- Move / Rotate / Scale gizmo on selection
    F                          -- frame selected object

Keys this script itself adds (non-conflicting):
    P     -- print current viewer camera pose, robot pose(s), and every
              task_object's pose (position, quaternion) -- for copying
              values (after nudging things with the Move/Rotate gizmo)
              back into the task config file
    K     -- dump the current physics state (serialized) and save it to the
              loaded task module's INIT_STATE_PATH, if it defines one --
              for authoring/refreshing a task's init_state pkl after
              hand-placing objects with the Move/Rotate gizmo
    C     -- call the loaded task module's task_completion_check(env), if it
              defines one, and print the result -- verifies e.g. an
              object_states.Inside placement actually reads True instead of
              eyeballing raw coordinates
    G     -- wake every task_object + robot. Do this after dragging something
              with the Move/Rotate gizmo and before SPACE: the gizmo edits the
              USD transform directly, not through the physics API, so a body
              that was already asleep will not respond to gravity/collisions
              on the next physics step until explicitly woken
    N     -- re-run task_mod.reset(env) on demand (the same call that runs
              once automatically at startup). Useful for tasks whose reset()
              tries to form a grasp (is_grasping() check + close-gripper
              fallback): nudge the target object near the gripper with the
              gizmo, press G to wake it, then N to retry the grasp attempt --
              without relaunching Isaac Sim. Not bound to R since R is Kit's
              native Scale-gizmo hotkey.
    SPACE -- toggle play/pause of physics stepping
    ESC   -- quit

Note on play/pause: Kit's own Timeline widget is visible in this layout but
does not drive OmniGibson's physics (which is stepped explicitly from
Python), so SPACE is the real play/pause control here, not the Timeline's
play button.
"""

from __future__ import annotations

import os
os.environ.setdefault("CARB_LOG_CHANNELS", "omni.physx.plugin=off")

import argparse
import pickle
import traceback

import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import KeyboardEventHandler

from damagesim.omnigibson.damageable_env import OGDamageableEnvironment
from scripts.teleop_b1k import TASK_REGISTRY, load_task_config, build_env_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task_name", type=str, default="default", choices=sorted(TASK_REGISTRY.keys()),
                   help="Task name (e.g. nav_to_table_02, carry_quilt, door_and_mug).")
    return p.parse_args()


_QUIT = [False]
_PAUSED = [True]  # start paused so the initial arrangement is easy to inspect


def main():
    args = parse_args()
    task_cfg, task_mod = load_task_config(args.task_name)

    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = task_cfg.enable_transition_rules

    env_config = build_env_config(task_cfg)
    env = OGDamageableEnvironment(configs=env_config)

    def run_task_reset():
        """Call task_mod.reset(env), if the task module defines one.
        Deliberately does NOT call env.reset() first: that would snap every
        object (including anything just gizmo-dragged into place, e.g. near
        the gripper for a grasp attempt) back to its raw TASK_OBJECTS spawn
        pose, undoing the hand-placement this key exists to iterate on."""
        if hasattr(task_mod, "reset") and callable(task_mod.reset):
            try:
                task_mod.reset(env)
                return True
            except Exception as e:
                print(f"[inspect_scene] task_mod.reset(env) raised {type(e).__name__}: {e}")
                traceback.print_exc()
                return False
        print("[inspect_scene] task module has no reset(env); nothing to run.")
        return False

    # Startup only: get a clean baseline (every object at its TASK_OBJECTS
    # spawn pose) before task_mod.reset() runs once. Subsequent N presses
    # deliberately skip this env.reset() -- see run_task_reset()'s docstring.
    env.reset()
    run_task_reset()

    if task_cfg.viewer_camera_pos is not None and task_cfg.viewer_camera_orn is not None:
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
            orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
        )

    # Deliberately NOT calling setup_viewport_layout() here -- that hides
    # Stage/Property/Timeline, which is exactly what this script exists to
    # keep visible (unlike scripts/teleop_b1k.py's recording-focused layout).

    def toggle_pause():
        _PAUSED[0] = not _PAUSED[0]
        print(f"[inspect_scene] {'PAUSED' if _PAUSED[0] else 'PLAYING'}")

    def request_quit():
        _QUIT[0] = True

    def _print_pose(label, pos, orn):
        print(f"[inspect_scene] {label} position:", pos.flatten().tolist()[:3])
        print(f"[inspect_scene] {label} orientation (x,y,z,w):", orn.flatten().tolist()[:4])

    def print_poses():
        """Camera + robot + every task_object's current position/orientation --
        for copying tuned values (after dragging things with the Move/Rotate
        gizmo) back into the task config file."""
        pos, orn = og.sim.viewer_camera.get_position_orientation()
        _print_pose("viewer_camera", pos, orn)

        for robot in env.robots:
            pos, orn = robot.get_position_orientation()
            _print_pose(f"robot '{robot.name}'", pos, orn)

        for obj_name in task_cfg.task_objects:
            obj = env.scene.object_registry("name", obj_name)
            if obj is None:
                continue
            pos, orn = obj.get_position_orientation()
            _print_pose(f"object '{obj_name}'", pos, orn)

    def save_init_state():
        init_state_path = getattr(task_mod, "INIT_STATE_PATH", None)
        if not init_state_path:
            print("[inspect_scene] task module has no INIT_STATE_PATH; nothing to save.")
            return
        os.makedirs(os.path.dirname(init_state_path), exist_ok=True)
        state = og.sim.dump_state(serialized=True)
        with open(init_state_path, "wb") as f:
            pickle.dump(state, f)
        print(f"[inspect_scene] saved current physics state to {init_state_path}")

    def check_completion():
        fn = getattr(task_mod, "task_completion_check", None)
        if not callable(fn):
            print("[inspect_scene] task module has no task_completion_check; nothing to check.")
            return
        try:
            result = fn(env)
        except Exception as e:
            print(f"[inspect_scene] task_completion_check(env) raised: {e}")
            return
        print(f"[inspect_scene] task_completion_check(env) -> {result}")

    def re_run_reset():
        """Re-run task_mod.reset(env) on demand (see docstring for the N key).
        Lets you nudge an object near the gripper with the gizmo, wake it (G),
        then retry the task's own grasp-forming logic without relaunching."""
        if run_task_reset():
            print("[inspect_scene] re-ran task_mod.reset(env).")

    def wake_all():
        """Force every task_object + robot to wake up. Dragging an object with
        the viewport Move/Rotate gizmo edits its USD transform directly, not
        through the physics API's set_position_orientation -- if the body was
        already asleep, PhysX keeps simulating it from its last-known (stale)
        pose and it will not respond to gravity/collisions until woken."""
        woken = 0
        for obj_name in task_cfg.task_objects:
            obj = env.scene.object_registry("name", obj_name)
            if obj is not None and hasattr(obj, "wake"):
                obj.wake()
                woken += 1
        for robot in env.robots:
            if hasattr(robot, "wake"):
                robot.wake()
                woken += 1
        print(f"[inspect_scene] woke {woken} object(s)/robot(s).")

    KeyboardEventHandler.initialize()
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.SPACE, callback_fn=toggle_pause,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE, callback_fn=request_quit,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.P, callback_fn=print_poses,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.K, callback_fn=save_init_state,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.C, callback_fn=check_completion,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.G, callback_fn=wake_all,
    )
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.N, callback_fn=re_run_reset,
    )

    print(f"[inspect_scene] task '{args.task_name}' loaded. SPACE=play/pause, ESC=quit, "
          f"P=print camera+robot+object poses, C=run task_completion_check, "
          f"G=wake all objects/robots (do this after gizmo-dragging something, before SPACE), "
          f"N=re-run task_mod.reset(env) on demand (e.g. to retry a grasp attempt).")
    print("[inspect_scene] starting PAUSED so the initial layout is easy to inspect.")

    try:
        while not _QUIT[0]:
            if _PAUSED[0]:
                og.sim.render()
            else:
                og.sim.step()
    finally:
        KeyboardEventHandler.reset()
        og.shutdown()


if __name__ == "__main__":
    main()
