# Scene inspector script (separate from teleop_b1k.py)

## Context

While authoring new tasks (`nav_to_table_02`, `carry_quilt`, `door_and_mug`), the user
hit two frictions with `scripts/teleop_b1k.py`'s GUI:

1. Camera positions had to be guessed offline and are often wrong on first load.
2. The current viewport-only layout (`damagesim/utils/misc_utils.py`'s
   `setup_viewport_layout()`, called from `teleop_b1k.py:894`) explicitly hides
   every non-Viewport Kit window (Stage tree, Property panel, Timeline), so the
   user can't inspect/nudge an object's Transform in the Property panel or use
   Play/Pause the way they normally would in Isaac Sim.

The user explicitly rejected editing `setup_viewport_layout()` / `teleop_b1k.py`
in place ("急に変えようとするのこわすぎ" — that function is load-bearing for the
existing, working teleop-collection flow across many tasks) and asked instead
for a **separate script** whose only job is: load a task's scene with the
normal full Isaac-Sim editor layout intact, so positions (robot spawn, camera,
task objects) can be eyeballed and tuned interactively before being copied back
into the task config `.py` files.

## Approach

Add a new script: **`scripts/inspect_scene.py`**. It reuses existing
lookup/config-building code from `scripts/teleop_b1k.py` via import (read-only
reuse — `teleop_b1k.py` itself is not modified) and skips everything
teleop/HDF5-specific:

- `TASK_REGISTRY`, `load_task_config`, `build_env_config`,
  `build_external_sensors_config` — imported from `scripts.teleop_b1k`
  (`teleop_b1k.py:68-88`, `:100-106`, `:169-184`). These already resolve a
  `--task_name` to a `TaskConfig` and assemble the `env`/`scene`/`robots`/
  `objects`/`task` cfg dict; no need to duplicate this logic.
- Set `gm.USE_GPU_DYNAMICS` / `gm.ENABLE_TRANSITION_RULES` from `task_cfg`
  **before** constructing the env (mirrors `teleop_b1k.py:850-851` — this
  ordering is required, macros configure the physics backend at Kit launch).
- Construct `OGDamageableEnvironment(configs=env_config)` directly (same class
  `teleop_b1k.py:867` uses) and call `env.reset()`. No
  `OGDamageableDataCollectionWrapper` — no HDF5 recording here.
- If the task module defines `reset(env)` (most task files do, e.g.
  `oopsiebench/envs/behavior1k/open_drawer.py:89`), call it once in a
  try/except after `env.reset()`, so task objects land in their intended
  starting arrangement (seated on tables/beds per each task's own reset logic)
  rather than their raw un-seated spawn pose. Our 3 draft tasks
  (`nav_to_table_02`, `carry_quilt`, `door_and_mug`) all set
  `INIT_STATE_PATH = None`, so this works today with no pkl file needed.
- Set the viewer camera to `task_cfg.viewer_camera_pos/orn` if present (mirrors
  `teleop_b1k.py:493-497`), as a starting point for further adjustment.
- **Do not call `setup_viewport_layout()`.** Per the Explore findings, the
  default Kit/Isaac-Sim layout (Stage, Property, Timeline, Content, Layer,
  Viewport) is what appears automatically as long as nothing hides it — this
  alone satisfies "let me see Stage/Property like normal Isaac Sim."
- Idle loop reusing the same pumping pattern as `teleop_b1k.py:157-166`'s
  `wait_for_enter()` (loop `og.sim.render()`/`og.sim.step()` — never
  block on `input()`/`breakpoint()` alone, since that can stall the Kit main
  loop and crash the renderer per that function's own docstring). Bind two
  simple keys via `KeyboardRobotController`'s or a raw `carb.input` keyboard
  interface (same registration style as `teleop_b1k.py:656-716`):
  - **SPACE** — toggle a local `PAUSED` flag: paused → loop calls only
    `og.sim.render()` (UI stays responsive, physics frozen); playing → loop
    also calls `og.sim.step()` each iteration. This is the actual "Play/Pause"
    the user asked for — note in the script's docstring that Kit's own
    Timeline widget will be *visible* but does not drive OmniGibson's physics
    (which is Python-stepped), so SPACE is the real control, not the Timeline
    play button.
  - **TAB** — print the current viewer camera position/orientation (reuse
    `teleop_b1k.py:270-277`'s `viewer_camera_print_pose_and_break` logic,
    without the `breakpoint()` — just print, so the render loop keeps
    running), for copying into a task file's `VIEWER_CAMERA_POS`/`ORN`.
  - **W/A/S/D/Q/E** — camera nudge, reusing `viewer_camera_nudge` logic
    (`teleop_b1k.py:280-305`) verbatim (small self-contained copy, ~25 lines,
    no import needed since it's a free function with no teleop-state deps).
  - **ESC** — break the loop, `og.shutdown()`, exit.
- CLI: just `--task_name` (default `"default"`), matching `teleop_b1k.py`'s
  own default and using the same `TASK_REGISTRY` so all existing tasks *and*
  the 3 new drafts are selectable immediately:
  ```
  python scripts/inspect_scene.py --task_name nav_to_table_02
  ```

## Files

- **New:** `scripts/inspect_scene.py` — the whole deliverable.
- **Unchanged:** `scripts/teleop_b1k.py`, `damagesim/utils/misc_utils.py`,
  `scripts/playback_b1k.py` — imported from, never edited.

## Verification

1. `python -m py_compile scripts/inspect_scene.py` — syntax sanity check
   (already the pattern used earlier for the 3 new task files, since a real
   GUI/Isaac-Sim run isn't feasible from this shell).
2. Report to the user that end-to-end verification (does Stage/Property
   actually show, does SPACE really pause physics, do the new task's objects
   land where expected) requires them to run it locally with a GUI —
   consistent with how the earlier camera-position/object-placement caveats
   in the 3 new task files were already flagged as "needs interactive
   verification."
