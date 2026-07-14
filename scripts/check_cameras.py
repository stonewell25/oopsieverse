#!/usr/bin/env python3
"""
Cheap render-verification gate for newly added EXTERNAL_CAMERA_CONFIGS entries,
to run BEFORE committing to a long multi-episode pointworld export batch.

Plays back ONE demo for a task (reusing the exact same env-construction path
as scripts/playback_b1k.py's run_playback) and dumps a PNG per external
camera from the first captured frame (right after scene.restore()), so a
human (or an image-reading review) can check that every camera actually sees
the workspace -- not a wall, not the ceiling, not clipped by furniture --
before spending hours re-exporting the full teleop demo set.

Usage:
    python scripts/check_cameras.py --task_name heat_saucepot \\
        --source_hdf5_path oopsiebench/demos/behavior1k/teleop/heat_saucepot_safe.hdf5 \\
        --out_dir camera_check
"""

from __future__ import annotations

import os
os.environ["CARB_LOG_CHANNELS"] = "omni.physx.plugin=off"

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch as th

import omnigibson as og
from omnigibson.macros import gm

from damagesim.omnigibson.damageable_env import OGDamageableDataPlaybackWrapper
from scripts.playback_b1k import (
    TASK_REGISTRY,
    attach_task_playback_hooks,
    build_external_sensors_config,
    load_task_module,
)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task_name", required=True, choices=sorted(TASK_REGISTRY.keys()))
    p.add_argument("--source_hdf5_path", required=True)
    p.add_argument("--out_dir", default="camera_check")
    p.add_argument("--episode_id", type=int, default=0)
    p.add_argument("--resolution", type=int, nargs=2, default=(256, 256), metavar=("HEIGHT", "WIDTH"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_mod = load_task_module(args.task_name)
    task_cfg = task_mod.get_task_config()

    gm.USE_GPU_DYNAMICS = task_cfg.use_gpu_dynamics
    gm.ENABLE_TRANSITION_RULES = False

    image_h, image_w = args.resolution
    external_sensors_config = build_external_sensors_config(
        task_cfg, task_cfg.robot_name, task_cfg.robot_type, image_h, image_w,
        modalities=["rgb", "seg_instance"],
    )
    robot_sensor_config = {
        "VisionSensor": {
            "modalities": ["rgb", "seg_instance"],
            "sensor_kwargs": {"image_height": image_h, "image_width": image_w},
        },
    }

    wrapper_cls = task_cfg.playback_wrapper_cls or OGDamageableDataPlaybackWrapper
    env = wrapper_cls.create_from_hdf5(
        input_path=args.source_hdf5_path,
        output_path=os.path.join(str(out_dir), f"{args.task_name}_camcheck_tmp.hdf5"),
        robot_obs_modalities=["proprio", "rgb", "seg_instance"],
        robot_sensor_config=robot_sensor_config,
        external_sensors_config=external_sensors_config,
        exclude_sensor_names=task_cfg.exclude_sensor_names,
        n_render_iterations=1,
        only_successes=False,
        activity_name=args.task_name,
    )

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor(task_cfg.viewer_camera_pos, dtype=th.float32),
        orientation=th.tensor(task_cfg.viewer_camera_orn, dtype=th.float32),
    )
    for _ in range(10):
        og.sim.step()

    if task_cfg.post_playback_env_setup is not None:
        task_cfg.post_playback_env_setup(env)

    attach_task_playback_hooks(env, task_cfg, task_mod)

    # Runs the full recorded episode (no cheap early-exit in playback_episode
    # today) but only ONE demo, purely to get scene.restore() to populate the
    # episode's objects before we grab a render -- acceptable one-off cost
    # ahead of a many-episode x many-camera batch export.
    env.playback_episode(episode_id=args.episode_id, record_data=True, save_images=True)

    camera_names = [f"external_sensor{name.split('_')[-1]}" for name in task_cfg.external_camera_configs]
    # current_traj_history gets flushed/cleared during playback_episode, so
    # grab the raw last-step obs instead (flatten_obs_space=True means it
    # already carries the same "external::<name>::rgb" keys). Camera poses
    # are static for the whole episode (parented to a stationary base_link,
    # or world_fixed), so the last frame frames the scene the same way the
    # first frame would.
    last_obs = env.current_obs
    saved = []
    for name in camera_names:
        rgb = last_obs.get(f"external::{name}::rgb")
        if rgb is None:
            print(f"[check_cameras] WARNING: no rgb captured for {name}")
            continue
        rgb_np = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        rgb_np = rgb_np[..., :3].astype(np.uint8)
        out_path = out_dir / f"{args.task_name}_{name}.png"
        cv2.imwrite(str(out_path), rgb_np[..., ::-1])  # RGB -> BGR for cv2
        saved.append(str(out_path))

    print(f"[check_cameras] saved {len(saved)} camera frames:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
