# Authoring a new BEHAVIOR-1K (OmniGibson) task

End-to-end workflow for adding a new `oopsiebench/envs/behavior1k/*.py` task:
scene selection → object selection/placement → physics verification → init-state
pkl → teleop data collection → playback.

## 1. Pick a scene

BEHAVIOR-1K ships 50 curated scenes. Browse look & per-room furniture lists on
Stanford's [Knowledgebase Dashboard](https://behavior.stanford.edu/knowledgebase/).

You can also inspect a scene's JSON directly without booting Isaac Sim, which is
much faster for a first pass:

```bash
python3 -c "
import json
d = json.load(open('externals/behavior1k/datasets/behavior-1k-assets/scenes/<SceneName>/json/<SceneName>_best.json'))
info = d['objects_info']['init_info']
rooms = set()
for o in info.values():
    rooms.update(o['args'].get('in_rooms') or [])
print(sorted(rooms))
"
```

List the furniture already present in a given room (useful for finding an
anchor piece to look up by name at runtime instead of spawning a stand-in --
see `BOOKCASE_NAME` / `BURNER_NAME` pattern in existing tasks):

```bash
python3 -c "
import json
d = json.load(open('externals/behavior1k/datasets/behavior-1k-assets/scenes/<SceneName>/json/<SceneName>_best.json'))
info = d['objects_info']['init_info']
room = '<room_instance_name>'
for name, o in info.items():
    if room in (o['args'].get('in_rooms') or []):
        print(name, o['args']['category'])
"
```

## 2. Pick objects

List available categories / models:

```python
from omnigibson.utils.asset_utils import get_all_object_categories, get_all_object_category_models
get_all_object_categories()
get_all_object_category_models("hardback")  # example category
```

Visually confirm a model before committing to it -- bbox numbers and texture
atlases are not reliable stand-ins for the actual mesh (e.g. one `hardback`
model, `csgebz`, looked like a wooden slat in-sim despite reasonable bbox
numbers):

```bash
python -m omnigibson.examples.objects.visualize_object
# run with no args to interactively pick category -> model and load it standalone
```

## 3. Create the task file

Copy an existing task module under `oopsiebench/envs/behavior1k/` (e.g.
`nav_shelf_clip.py`) as a template. Edit `get_task_config()`'s `scene_config`
(`scene_model`, `load_room_instances`), `robot_config`, and `TASK_OBJECTS`.

Register the new task name in `scripts/teleop_b1k.py`'s `TASK_REGISTRY` dict
(~line 68):

```python
"your_task_name": "your_task_name",
```

## 4. Place objects & verify physics

```bash
python scripts/inspect_scene.py --task_name your_task_name
```

- Starts **PAUSED**. Click an object in the viewport, use native Kit gizmos
  `W` (move) / `E` (rotate) to hand-place it.
- **G** -- wake every task_object + robot. Required after a gizmo edit: the
  gizmo writes the USD transform directly, not through the physics API, so a
  sleeping body won't react to gravity/collisions on the next step until
  woken.
- **SPACE** -- toggle physics play/pause. Confirm nothing falls through the
  floor / clips into a wall / rolls off the intended surface.
- **P** -- print current viewer camera pose, robot pose(s), and every
  task_object's pose. Copy these back into `TASK_OBJECTS` (and
  `robot_config["position"/"orientation"]`) in the task file.

If an object lands somewhere unexpected (floor, wall, wrong furniture), the
usual culprit is x/y being outside the footprint of the surface it's meant to
rest on -- compare against objects that *did* settle correctly rather than
just nudging z.

## 5. Save the init-state pkl

Once the layout is stable and reproducible, **K** dumps the current physics
state (serialized) to the task module's `INIT_STATE_PATH`. **C** runs
`task_completion_check(env)` if defined -- verify it can actually flip to
`True` (e.g. nudge the target object fully into place) before trusting the
saved state for real data collection.

## 6. Collect teleop demos

```bash
python scripts/teleop_b1k.py --task_name your_task_name \
  --collect_hdf5_path demos/behavior1k/teleop_data/your_task_name.hdf5 --n_episodes 5
```

Keys during teleop: `K` end episode, `ENTER` start/confirm, `ESC` quit (saves
completed episodes), `BACKSPACE` discard current trajectory, `TAB` print
camera pose.

## 7. Playback / visualize / metrics

```bash
python scripts/playback_b1k.py --task_name your_task_name \
  --source_hdf5_path demos/behavior1k/teleop_data/your_task_name.hdf5 \
  --playback_hdf5_path demos/behavior1k/playback_data/your_task_name.hdf5 \
  --playback --visualize --compute_metrics
```
