# Teleop Tutorial (`scripts/teleop_b1k.py`)

Keyboard teleop for Behavior1k-style OmniGibson tasks. This doc reflects the
script's actual runtime behavior, which in a couple of places diverges from
its own module docstring (noted below).

## 1. Running it

```bash
# Simple teleop, 1 episode, saves a video on quit
python scripts/teleop_b1k.py --task_name shelve_item --save_video

# Collect 5 episodes to HDF5 for later playback
python scripts/teleop_b1k.py --task_name shelve_item \
    --collect_hdf5_path demos/behavior1k/teleop_data/shelve_item.hdf5 --n_episodes 5
```

Available `--task_name` values: see `TASK_REGISTRY` in the script (e.g.
`shelve_item`, `add_firewood`, `pour_water`, `open_drawer`, `wipe_counter`,
`nav_to_table`, `pick_egg`, `fill_bowl`, `place_plate`, `turn_on_faucet`,
`heat_saucepot`, `open_single_door`, `food_in_microwave`, `default`).

Other useful flags:
- `--live_feedback` — show live health bars / object coloring during teleop
- `--save_obs_to_hdf5` — also save image observations during teleop (normally only saved during playback)
- `--skip_hdf5_save` — don't write the HDF5 at all
- `--teleop_device spacemouse` — use a SpaceMouse instead of the keyboard

## 2. What happens on launch

1. The env is built from the task config and reset to its initial state (from
   a saved init-state pickle if one exists for the task).
2. The viewer window opens.
3. Before every episode, execution stops at a Python debugger `breakpoint()`.
   You'll see `Press c to continue` printed — **type `c` and press Enter in
   the terminal** to actually start the episode. This happens again after
   each episode completes, so expect to type `c` repeatedly through a session.

**Click the viewer window first** so it has keyboard focus before you try to
move anything.

## 3. Controlling the robot

These all come from OmniGibson's `KeyboardRobotController` (not this script) —
**note this is the single-active-arm class, not `BimanualKeyboardRobotController`**
(which also exists in `omnigibson/utils/ui_utils.py` but `teleop_b1k.py` never
imports/uses it, for any robot). That distinction matters for Tiago (3b).

### 3a. Single fixed-arm robots (Franka tasks: `open_drawer`, `pick_egg`, `place_plate`, etc.)

| Keys | Action |
|---|---|
| Arrow keys | Move arm end-effector along x / y |
| `p` / `;` | Move arm along z |
| `n` / `b` | Rotate arm about x |
| `o` / `u` | Rotate arm about y |
| `v` / `c` | Rotate arm about z |
| `t` | Toggle gripper open/close |
| `1` / `2` | Select previous/next joint (only matters if the robot has other `JointController`-driven DOFs to jog) |
| `[` / `]` | Jog selected joint backward/forward |
| `m` | Render onboard sensor modalities |

These robots have no base controller at all (fixed base), so there's no base
movement here.

### 3b. Tiago — mobile, two arms (`nav_to_table`, `nav_to_table_02`, `door_and_mug`)

Tiago has **two IK arms (`arm_left`/`arm_right`) and two grippers
(`gripper_left`/`gripper_right`), but only one arm and one gripper are
"active" (controllable) at a time** — the arrow-keys/`p;`/`nb`/`ou`/`vc`
group and the `t` toggle always act on whichever arm/gripper is currently
active, switched with dedicated keys:

| Keys | Action |
|---|---|
| `3` / `4` | Switch the **active arm** (`arm_left` ↔ `arm_right`) that the arrow-keys/`p;`/`nb`/`ou`/`vc` group controls. Prints `Now controlling arm <name> EEF`. |
| Arrow keys, `p`/`;`, `n`/`b`, `o`/`u`, `v`/`c` | Move/rotate the **currently active arm's** end-effector (same axis mapping as 3a). |
| `5` / `6` | Switch the **active gripper** (`gripper_left` ↔ `gripper_right`) that `t` toggles. Prints `Now controlling gripper <name> with binary toggling`. |
| `t` | Toggle the **currently active gripper** open/close. |
| `1` / `2` | Select previous/next DOF among **all `JointController`/`HolonomicBaseJointController`-driven DOFs** — this is how the **base** (holonomic x/y/yaw) is selected, plus e.g. trunk lift / head pan-tilt if present. |
| `[` / `]` | Jog the currently selected DOF (from `1`/`2`) backward/forward — **this is how you actually drive the base**, not `i`/`k`/`l`/`j`. |
| `m` | Render onboard sensor modalities |

**Gotcha:** an earlier version of this doc listed `i`/`k` (turn) and `l`/`j`
(forward/back) for base movement. That's only correct for a robot using
`DifferentialDriveController`. Tiago's base controller is
`HolonomicBaseJointController` (`omnigibson/robots/tiago.py`), which falls
into the generic `1`/`2`-select + `[`/`]`-jog path above instead — there is
no continuous WASD-style base driving for Tiago in this repo, you select the
base's x/y/yaw DOF with `1`/`2` and nudge it with `[`/`]` like any other
joint.

## 4. Session keys added by this script

| Key | Action |
|---|---|
| `ESC` | Quit. Saves completed episodes to HDF5 (discards any in-progress episode). |
| `K` | End the current episode (triggers reset, starts the next one). |
| `BACKSPACE` | Discard the current trajectory and start over — no save, doesn't count toward `--n_episodes`. |
| `TAB` | Debug tool: prints the viewer camera's position/orientation, then drops into a `breakpoint()`. |
| `W` / `A` / `D` | Move the **viewer camera** (not the robot) forward/left/right. |
| `Q` / `E` | Move the **viewer camera** down/up (world Z). |

### Known discrepancies vs. the module docstring

The script's own header comment says `TAB` ends an episode and `S` saves
serialized state. That's not what currently happens:

- **`TAB`** is actually bound to the "print camera pose + breakpoint" debug
  action. **`K`** is what ends the episode.
- **`S`** is registered twice — once for "save serialized state" and again
  for "move viewer camera backward." The second registration silently wins,
  so pressing `S` only moves the camera; the save-state feature is currently
  unreachable from the keyboard.

If you need the save-state feature, rebind one of the two `S` handlers to a
free key.

## 5. Ending a session

- If collecting demos (`--collect_hdf5_path` set, and `--skip_hdf5_save` not
  passed), the HDF5 file is written on quit/completion.
- If `--save_video` is set, an MP4 (plus force/temperature plots if
  configured for the task) is written to `demos/behavior1k/teleop_videos/`.
