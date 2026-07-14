"""
Exports a played-back OG episode to a single .npz file consumable by the
PointWorld bridge (see ~/PointWorld/oopsieverse_bridge/).

Interchange is npz (not HDF5) because the HDF5 build used by this repo's
conda env is not guaranteed compatible with every downstream consumer, and
npz keeps the two repos fully decoupled.

Usage: attach a ``PointWorldRecorder`` to an ``OGDamageableDataPlaybackWrapper``
instance as ``env.pointworld_recorder`` before calling ``playback_dataset``;
``playback_episode`` calls ``on_step`` once per played-back step when the
attribute is set (see damageable_env.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch as th
except ImportError:
    th = None

try:
    import omnigibson as og
except ImportError:
    og = None


def _to_numpy(x):
    if th is not None and isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# PointWorld eval domain whose training embodiment each OG robot class most
# resembles: single fixed-base arm -> "droid" (Panda/DROID), mobile bimanual ->
# "behavior" (R1Pro/BEHAVIOR-1K). Consumed by ~/PointWorld/oopsieverse_bridge
# to validate the --domains choice against the episode's robot.
_ROBOT_TYPE_TO_POINTWORLD_DOMAIN = {
    "FrankaPanda": "droid",
    "FrankaMounted": "droid",
    "Franka": "droid",
    "Tiago": "behavior",
    "R1": "behavior",
    "R1Pro": "behavior",
}


def _opengl_pose_to_cv_world2cam(position, orientation_xyzw):
    """
    OG/USD cameras look down -Z with +Y up (OpenGL convention). PointWorld's
    unprojection (mirrored from PointWorld's project_depth_to_world) assumes
    a camera frame with +Z forward / +Y down (OpenCV convention).

    Returns the 4x4 world->camera extrinsic matrix in OpenCV convention.
    """
    import omnigibson.utils.transform_utils as T

    R_gl = _to_numpy(T.quat2mat(orientation_xyzw))
    t = _to_numpy(position)
    flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    R_cv = R_gl @ flip

    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, :3] = R_cv
    cam2world[:3, 3] = t
    world2cam = np.linalg.inv(cam2world).astype(np.float32)
    return world2cam


class PointWorldRecorder:
    """Accumulates per-step robot/object/camera/health data during OG playback."""

    def __init__(
        self,
        env,
        out_dir: str,
        task_name: str,
        stride: int = 3,
        camera_names: Optional[List[str]] = None,
        joints_only: bool = False,
        source_tag: Optional[str] = None,
        checkpoint_every: Optional[int] = None,
    ):
        self.env = env
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name
        self.stride = max(1, int(stride))
        self.joints_only = joints_only
        # If set, dump the buffered data to the final .npz path (overwriting
        # it) every N *recorded* frames. Playback can crash mid-episode from
        # unbounded native-side (Isaac Sim) memory growth as steps pile up --
        # not at any predictable step -- so periodic checkpointing is the only
        # way to guarantee a usable partial export survives a hard crash that
        # Python can't catch (SIGSEGV).
        self.checkpoint_every = checkpoint_every
        # Distinguishes output filenames when the same task/demo_id is
        # exported from different source HDF5s (e.g. shelve_item's safe vs.
        # unsafe teleop demo are both task_name="shelve_item" demo_id=0, and
        # would otherwise silently overwrite each other's export).
        self.source_tag = source_tag or task_name
        self._camera_names_override = camera_names

        self._episode_id: Optional[int] = None
        self._camera_names: List[str] = []
        self._cam_intrinsic: Dict[str, np.ndarray] = {}
        self._cam_extrinsic: Dict[str, np.ndarray] = {}
        self._object_names: List[str] = []
        self._object_masses: List[float] = []
        self._reset_buffers()

    # ── lifecycle ──────────────────────────────────────────────────────

    def _reset_buffers(self):
        self._buf_joint_positions: List[np.ndarray] = []
        self._buf_gripper_width: List[float] = []
        self._buf_eef_pose: List[np.ndarray] = []
        self._buf_object_poses: List[np.ndarray] = []
        self._buf_health: List[np.ndarray] = []
        self._buf_damage_info: List[str] = []
        self._buf_cam_rgb: Dict[str, List[np.ndarray]] = {}
        self._buf_cam_depth: Dict[str, List[np.ndarray]] = {}
        self._buf_cam_seg: Dict[str, List[np.ndarray]] = {}
        self._seg_id_map: Dict[str, Dict[int, str]] = {}
        self._step_counter = 0
        self._joint_names: Optional[List[str]] = None
        self._health_link_names: Optional[List[str]] = None
        self._robot_base_pose: Optional[np.ndarray] = None
        self._robot_type: str = "unknown"
        self._recommended_domain: str = "unknown"
        self._scene_state_captured = False

    def on_episode_start(self, episode_id: int):
        self._episode_id = episode_id
        self._reset_buffers()

    def _capture_scene_state(self):
        """Snapshot robot joints / scene objects / camera intrinsics-extrinsics.

        Must run *after* ``playback_episode``'s ``scene.restore(...)`` has
        populated the episode-specific interactive objects (book, wineglass,
        etc.) -- those don't exist yet when ``on_episode_start`` fires, so
        this is deferred to the first ``on_step`` call instead.
        """
        scene = self.env.scene
        robot = scene.robots[0]
        self._joint_names = list(robot.joint_names) if hasattr(robot, "joint_names") else list(robot.joints.keys())

        self._robot_type = type(robot).__name__
        # Runtime robot classes are Damageable-wrapped (DamageableFrankaPanda,
        # DamageableTiago, ...) -- strip that prefix so the lookup above
        # actually matches instead of always missing into "unknown".
        domain_lookup_key = self._robot_type.removeprefix("Damageable")
        self._recommended_domain = _ROBOT_TYPE_TO_POINTWORLD_DOMAIN.get(domain_lookup_key, "unknown")
        print(f"[pointworld_export] robot_type={self._robot_type} -> recommended PointWorld domain: {self._recommended_domain}")

        base_pos, base_orn = robot.get_position_orientation(frame="world")
        self._robot_base_pose = np.concatenate([_to_numpy(base_pos), _to_numpy(base_orn)]).astype(np.float32)

        # Non-robot scene objects, sorted for a stable index.
        self._object_names = sorted(
            o.name for o in scene.objects if o not in scene.robots
        )
        self._object_masses = []
        for name in self._object_names:
            obj = scene.object_registry("name", name)
            try:
                mass = float(sum(link.mass for link in obj.links.values()))
            except Exception:
                mass = float("nan")
            self._object_masses.append(mass)

        if not self.joints_only:
            external_sensors = getattr(self.env, "_external_sensors", None) or {}
            if self._camera_names_override is not None:
                self._camera_names = [n for n in self._camera_names_override if n in external_sensors]
            else:
                self._camera_names = sorted(external_sensors.keys())

            self._cam_intrinsic = {}
            self._cam_extrinsic = {}
            for name in self._camera_names:
                sensor = external_sensors[name]
                K = _to_numpy(sensor.intrinsic_matrix).astype(np.float32)
                # Isaac Sim's camera-projection annotator can lag by a frame or two
                # right after scene setup; a degenerate (all-zero fx/fy) matrix
                # means it hasn't populated yet, so nudge the sim and retry.
                from omnigibson.sensors.vision_sensor import render as _og_render

                retries = 0
                while (K[0, 0] <= 0 or K[1, 1] <= 0) and retries < 10:
                    _og_render()
                    K = _to_numpy(sensor.intrinsic_matrix).astype(np.float32)
                    retries += 1
                if K[0, 0] <= 0 or K[1, 1] <= 0:
                    print(f"[pointworld_export] WARNING: {name} intrinsic_matrix still degenerate after retries: {K}")
                pos, orn = sensor.get_position_orientation(frame="world")
                E = _opengl_pose_to_cv_world2cam(pos, orn)
                self._cam_intrinsic[name] = K
                self._cam_extrinsic[name] = E
                self._buf_cam_rgb[name] = []
                self._buf_cam_depth[name] = []
                self._buf_cam_seg[name] = []
                self._seg_id_map[name] = {}

        self._scene_state_captured = True

    def on_step(self, step_index: int, obs: dict, info: dict):
        if not self._scene_state_captured:
            self._capture_scene_state()

        self._step_counter += 1
        if (self._step_counter - 1) % self.stride != 0:
            return

        scene = self.env.scene
        robot = scene.robots[0]

        self._buf_joint_positions.append(_to_numpy(robot.get_joint_positions()).astype(np.float32))

        finger_names = robot.finger_joint_names[robot.default_arm]
        joint_pos_by_name = dict(zip(self._joint_names, _to_numpy(robot.get_joint_positions())))
        width = float(sum(joint_pos_by_name[n] for n in finger_names if n in joint_pos_by_name))
        self._buf_gripper_width.append(width)

        eef_pos = _to_numpy(obs.get("eef_pos"))
        eef_ori = _to_numpy(obs.get("eef_ori"))
        self._buf_eef_pose.append(np.concatenate([eef_pos, eef_ori]).astype(np.float32))

        poses = []
        for name in self._object_names:
            obj = scene.object_registry("name", name)
            pos, orn = obj.get_position_orientation(frame="world")
            poses.append(np.concatenate([_to_numpy(pos), _to_numpy(orn)]).astype(np.float32))
        self._buf_object_poses.append(np.stack(poses, axis=0) if poses else np.zeros((0, 7), np.float32))

        health = obs.get("health")
        self._buf_health.append(_to_numpy(health).astype(np.float32) if health is not None else np.zeros(0, np.float32))
        if self._health_link_names is None:
            self._health_link_names = list(getattr(self.env, "health_list_link_names", None) or [])

        damage_info = info.get("damage_info") if info is not None else None
        self._buf_damage_info.append(json.dumps(damage_info, default=str) if damage_info is not None else "{}")

        if self.joints_only:
            if self.checkpoint_every and len(self._buf_joint_positions) % self.checkpoint_every == 0:
                self._write_npz(partial=True)
            return

        obs_info = (info.get("obs_info") or {}) if info is not None else {}
        external_info = obs_info.get("external") or {}
        for name in self._camera_names:
            rgb = obs.get(f"external::{name}::rgb")
            depth = obs.get(f"external::{name}::depth_linear")
            seg = obs.get(f"external::{name}::seg_instance")
            seg_info = external_info.get(name, {}).get("seg_instance")
            if seg_info:
                self._seg_id_map[name].update({int(k): str(v) for k, v in seg_info.items()})
            self._buf_cam_rgb[name].append(_to_numpy(rgb)[..., :3].astype(np.uint8) if rgb is not None else None)
            if depth is not None:
                depth_mm = np.clip(_to_numpy(depth) * 1000.0, 0, 65535).astype(np.uint16)
            else:
                depth_mm = None
            self._buf_cam_depth[name].append(depth_mm)
            self._buf_cam_seg[name].append(_to_numpy(seg).astype(np.int32) if seg is not None else None)

        if self.checkpoint_every and len(self._buf_joint_positions) % self.checkpoint_every == 0:
            self._write_npz(partial=True)

    def finalize(self) -> str:
        return self._write_npz(partial=False)

    def _write_npz(self, partial: bool) -> str:
        out = {
            "task_name": self.task_name,
            "demo_id": int(self._episode_id) if self._episode_id is not None else -1,
            "robot_name": self.env.scene.robots[0].name,
            "robot_type": self._robot_type,
            "recommended_domain": self._recommended_domain,
            "robot_base_pose": self._robot_base_pose if self._robot_base_pose is not None else np.zeros(7, np.float32),
            "stride": self.stride,
            "time_steps": len(self._buf_joint_positions),
            "joint_names": np.array(self._joint_names, dtype=object),
            "joint_positions": np.stack(self._buf_joint_positions, axis=0)
            if self._buf_joint_positions
            else np.zeros((0, 0), np.float32),
            "gripper_width": np.array(self._buf_gripper_width, dtype=np.float32),
            "eef_pose": np.stack(self._buf_eef_pose, axis=0) if self._buf_eef_pose else np.zeros((0, 7), np.float32),
            "object_names": np.array(self._object_names, dtype=object),
            "object_masses": np.array(self._object_masses, dtype=np.float32),
            "object_poses": np.stack(self._buf_object_poses, axis=0)
            if self._buf_object_poses
            else np.zeros((0, 0, 7), np.float32),
            "health_link_names": np.array(self._health_link_names or [], dtype=object),
            "health": np.stack(self._buf_health, axis=0) if self._buf_health else np.zeros((0, 0), np.float32),
            "damage_info": np.array(self._buf_damage_info, dtype=object),
            "partial": partial,
        }

        for name in self._camera_names:
            rgb_list = self._buf_cam_rgb[name]
            depth_list = self._buf_cam_depth[name]
            seg_list = self._buf_cam_seg[name]
            if rgb_list and rgb_list[0] is not None:
                out[f"{name}_rgb"] = np.stack(rgb_list, axis=0)
            if depth_list and depth_list[0] is not None:
                out[f"{name}_depth"] = np.stack(depth_list, axis=0)
            if seg_list and seg_list[0] is not None:
                out[f"{name}_seg_instance"] = np.stack(seg_list, axis=0)
            out[f"{name}_intrinsic"] = self._cam_intrinsic[name]
            out[f"{name}_extrinsic"] = self._cam_extrinsic[name]
            out[f"{name}_seg_id_map"] = json.dumps(self._seg_id_map.get(name, {}))

        suffix = "_joints" if self.joints_only else ""
        out_path = self.out_dir / f"episode_{self.source_tag}_{self._episode_id}{suffix}.npz"
        # Write to a tmp file then rename, so a crash mid-write (e.g. the same
        # native segfault this checkpointing exists to survive) can't leave a
        # half-written/corrupt .npz at out_path -- os.replace is atomic on the
        # same filesystem.
        # Must itself end in ".npz" -- np.savez_compressed silently appends
        # ".npz" to any path that doesn't already end with it, so a name like
        # "*.npz.tmp" actually gets written to "*.npz.tmp.npz" and the
        # rename below then fails to find "*.npz.tmp".
        tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
        np.savez_compressed(tmp_path, **out)
        tmp_path.replace(out_path)
        label = f"checkpoint ({out['time_steps']} steps)" if partial else f"final ({out['time_steps']} steps)"
        print(f"[pointworld_export] wrote {label} → {out_path}")
        return str(out_path)
