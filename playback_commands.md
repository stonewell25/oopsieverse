# 残り playback コマンド一覧（コピペ用）

前提: `conda activate oopsieverse_b1k` 済み、cwd = `/home/sena/oopsieverse`。
`OMNIGIBSON_HEADLESS=1` を毎回頭に付けてheadless実行（GUI起動を避ける想定）。

完了済み（スキップ）: `heat_saucepot_safe`, `fill_bowl_safe`, `fill_bowl_unsafe`

---

### add_firewood_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name add_firewood \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/add_firewood_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/add_firewood_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### add_firewood_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name add_firewood \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/add_firewood_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/add_firewood_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### food_in_microwave_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name food_in_microwave \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/food_in_microwave_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/food_in_microwave_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### food_in_microwave_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name food_in_microwave \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/food_in_microwave_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/food_in_microwave_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### heat_saucepot_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name heat_saucepot \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/heat_saucepot_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/heat_saucepot_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### nav_to_table_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name nav_to_table \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/nav_to_table_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/nav_to_table_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### nav_to_table_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name nav_to_table \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/nav_to_table_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/nav_to_table_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### open_drawer_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name open_drawer \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/open_drawer_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/open_drawer_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### open_drawer_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name open_drawer \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/open_drawer_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/open_drawer_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### open_single_door_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name open_single_door \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/open_single_door_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/open_single_door_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### open_single_door_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name open_single_door \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/open_single_door_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/open_single_door_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### pick_egg_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name pick_egg \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/pick_egg_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/pick_egg_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### pick_egg_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name pick_egg \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/pick_egg_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/pick_egg_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### place_plate_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name place_plate \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/place_plate_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/place_plate_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### place_plate_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name place_plate \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/place_plate_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/place_plate_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### pour_water_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name pour_water \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/pour_water_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/pour_water_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### pour_water_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name pour_water \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/pour_water_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/pour_water_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### shelve_item_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name shelve_item \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/shelve_item_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/shelve_item_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### shelve_item_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name shelve_item \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/shelve_item_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/shelve_item_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### turn_on_faucet_safe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name turn_on_faucet \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/turn_on_faucet_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/turn_on_faucet_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### turn_on_faucet_unsafe: done
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name turn_on_faucet \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/turn_on_faucet_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/turn_on_faucet_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### wipe_counter_safe
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name wipe_counter \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/wipe_counter_safe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/wipe_counter_safe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

### wipe_counter_unsafe
```bash
OMNIGIBSON_HEADLESS=1 python scripts/playback_b1k.py \
  --task_name wipe_counter \
  --playback \
  --source_hdf5_path oopsiebench/demos/behavior1k/teleop/wipe_counter_unsafe.hdf5 \
  --playback_hdf5_path logs/pointworld_export_tmp_playback/wipe_counter_unsafe_playback.hdf5 \
  --export_pointworld_dir ~/oopsieverse/exports/pointworld_v2 \
  --export_checkpoint_every 50
```

---

## towel_fire は別枠（要確認、上のリストには含めていません）

`demos/behavior1k/teleop_data/towel_fire.hdf5` を確認したところ **`n_episodes=0`、`data`配下のdemoキーも0件**でした。中身が空で、teleopの録画（`teleop_videos/`にmp4は複数ある）はあるのに、それがこのhdf5に反映されていません。このファイルのままplaybackしても何も出力されないはずなので、他のタスクを先に流しつつ、towel_fireは元の収録データ（別ファイル or 再収録）を探す/作る必要があります。
