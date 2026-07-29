#!/usr/bin/env python3
"""
Dumps the RGB frames already baked into pointworld_export.py's npz exports
(`{cam}_rgb`, shape (T,H,W,3)) as PNGs, across many episodes at once.

Unlike scripts/check_cameras.py (which replays the full OmniGibson simulation
via HDF5 just to grab ONE frame per camera, for verifying newly-added camera
configs BEFORE a batch export), this reads only the already-exported npz --
no OmniGibson/Isaac Sim dependency, no simulation replay. It's for visually
inspecting what was actually captured across the whole episode (weird
occlusions, clipping, framing) across every task at once.

Usage
-----
    # every episode in the default export dir, every camera, whole episode
    python scripts/dump_camera_frames.py --coverage max

    # just the first half of two specific episodes, denser sampling
    python scripts/dump_camera_frames.py \\
        --npz_glob "episode_nav_to_table*.npz" --coverage half --stride 3

    # continuous playback instead of scrubbing PNGs: also write an mp4 per
    # episode/camera (add --no_frames to skip the PNGs and only get videos)
    python scripts/dump_camera_frames.py --video --video_fps 10 --no_frames
"""

from __future__ import annotations

import argparse
import glob as glob_module
import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_NPZ_DIR = os.path.expanduser("~/oopsieverse/exports/pointworld_v2")


def _camera_names(npz) -> list[str]:
    return sorted({k.rsplit("_rgb", 1)[0] for k in npz.files if k.endswith("_rgb")})


def dump_episode(
    npz_path: Path, out_dir: Path, coverage: str, stride: int, cameras: list[str] | None,
    write_frames: bool, write_video: bool, video_fps: float,
) -> int:
    d = np.load(npz_path, allow_pickle=True)
    cam_names = _camera_names(d)
    if not cam_names:
        print(f"[skip] {npz_path.name}: no camera RGB (joints-only export)")
        return 0
    if cameras:
        cam_names = [c for c in cam_names if c in cameras]

    time_steps = int(d["time_steps"])
    end_t = time_steps if coverage == "max" else max(1, time_steps // 2)

    episode_tag = npz_path.stem  # e.g. "episode_nav_to_table_02_0"
    episode_dir = out_dir / episode_tag
    n_saved = 0
    for cam in cam_names:
        rgb = d[f"{cam}_rgb"]  # (T, H, W, 3) uint8, RGB order
        frame_idxs = list(range(0, min(end_t, rgb.shape[0]), stride))

        if write_frames:
            cam_out = episode_dir / cam
            cam_out.mkdir(parents=True, exist_ok=True)
            for t in frame_idxs:
                cv2.imwrite(str(cam_out / f"frame_{t:04d}.png"), rgb[t][..., ::-1])
                n_saved += 1

        if write_video and frame_idxs:
            episode_dir.mkdir(parents=True, exist_ok=True)
            h, w = rgb.shape[1:3]
            video_path = episode_dir / f"{cam}.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h)
            )
            for t in frame_idxs:
                writer.write(rgb[t][..., ::-1])
            writer.release()
            print(f"       video: {video_path} ({len(frame_idxs)} frames @ {video_fps}fps)")

    print(f"[ok] {npz_path.name}: {len(cam_names)} camera(s), frames 0..{end_t - 1} (stride {stride}) "
          f"-> {n_saved} images in {episode_dir}" if write_frames else
          f"[ok] {npz_path.name}: {len(cam_names)} camera(s), frames 0..{end_t - 1} (stride {stride})")
    return n_saved


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--npz_dir", default=DEFAULT_NPZ_DIR,
                    help="Directory of pointworld_export.py npz exports to scan.")
    p.add_argument("--npz_glob", default="episode_*.npz",
                    help="Glob selecting which episodes to dump: relative to --npz_dir "
                         "(e.g. 'episode_nav_to_table*.npz'), or a full absolute path/pattern "
                         "(e.g. '/home/.../episode_nav_to_table_unsafe_0.npz'), in which case "
                         "--npz_dir is ignored.")
    p.add_argument("--out_dir", default="camera_check_frames")
    p.add_argument("--coverage", choices=["max", "half"], default="max",
                    help="'max': dump the whole episode. 'half': only the first half of "
                         "recorded (exported-stride) frames.")
    p.add_argument("--stride", type=int, default=10,
                    help="Save every Nth exported frame (units are pointworld_export.py's "
                         "own --stride-subsampled frames, not raw sim steps). Smaller = denser "
                         "but many more files.")
    p.add_argument("--cameras", nargs="*", default=None,
                    help="Only dump these camera names (e.g. external_sensor0). Default: all.")
    p.add_argument("--video", action="store_true",
                    help="Also write a <camera>.mp4 per episode/camera from the sampled frames, "
                         "so you can just play it back instead of scrubbing PNGs.")
    p.add_argument("--video_fps", type=float, default=10.0,
                    help="Playback fps for --video output. This is a *display* rate, unrelated "
                         "to the sim's real timestep -- --stride already controls how much real "
                         "time each saved frame represents.")
    p.add_argument("--no_frames", action="store_true",
                    help="Skip writing individual PNGs (use with --video if you only want the "
                         "video, to save disk/time).")
    args = p.parse_args()

    if args.no_frames and not args.video:
        raise SystemExit("--no_frames without --video would write nothing; drop --no_frames or add --video.")

    npz_dir = Path(os.path.expanduser(args.npz_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.path.isabs(args.npz_glob):
        npz_paths = sorted(Path(p) for p in glob_module.glob(args.npz_glob))
        search_desc = args.npz_glob
    else:
        npz_paths = sorted(npz_dir.glob(args.npz_glob))
        search_desc = f"{args.npz_glob} in {npz_dir}"
    if not npz_paths:
        raise SystemExit(f"No files matching {search_desc}")

    total = 0
    for npz_path in npz_paths:
        total += dump_episode(
            npz_path, out_dir, args.coverage, args.stride, args.cameras,
            write_frames=not args.no_frames, write_video=args.video, video_fps=args.video_fps,
        )

    print(f"\n[done] {len(npz_paths)} episode(s), {total} image(s) total -> {out_dir}")


if __name__ == "__main__":
    main()
