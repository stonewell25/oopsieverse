#!/usr/bin/env bash
# Replay every safe/unsafe teleop HDF5 under oopsiebench/demos/behavior1k/teleop/
# and export each episode to a PointWorld-bridge .npz under
# ~/oopsieverse/exports/pointworld_v2 (new directory -- not mixed with the old
# single-camera npz exports).
#
# Resumable: writes a per-source sentinel file under $STATUS_DIR once a source
# HDF5's export is verified complete; reruns skip any source that already has
# one. Each source gets up to $MAX_ATTEMPTS fresh-process attempts, since the
# very first Isaac Sim robot-init step occasionally hits a flaky numba/llvmlite
# JIT-compile race (SIGABRT/SIGSEGV before any data is written) that a plain
# retry with a new process reliably clears.
#
# Usage:
#   bash scripts/batch_playback_pointworld.sh
#
# Requires conda env: oopsieverse_b1k

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TELEOP_DIR="${TELEOP_DIR:-${REPO_ROOT}/oopsiebench/demos/behavior1k/teleop}"
EXPORT_DIR="${EXPORT_DIR:-${HOME}/oopsieverse/exports/pointworld_v2}"
STATUS_DIR="${REPO_ROOT}/logs/pointworld_export_status"
TMP_PLAYBACK_DIR="${REPO_ROOT}/logs/pointworld_export_tmp_playback"
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

mkdir -p "$EXPORT_DIR" "$STATUS_DIR" "$TMP_PLAYBACK_DIR" "${REPO_ROOT}/logs"

if ! command -v conda &>/dev/null; then
  # shellcheck source=/dev/null
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
fi
conda activate oopsieverse_b1k

export OMNIGIBSON_HEADLESS=1

FAILED=()

# Verifies that every expected episode for a source HDF5 has a non-partial
# .npz in EXPORT_DIR. Prints "OK" or "MISSING:<n_missing>" on stdout.
verify_export() {
  local source_tag="$1"
  python3 - "$EXPORT_DIR" "$source_tag" "$2" <<'PYEOF'
import sys, os
import h5py
import numpy as np

export_dir, source_tag, hdf5_path = sys.argv[1], sys.argv[2], sys.argv[3]
with h5py.File(hdf5_path, "r") as f:
    n_episodes = int(f["data"].attrs.get("n_episodes", 1))

missing = 0
for episode_id in range(n_episodes):
    npz_path = os.path.join(export_dir, f"episode_{source_tag}_{episode_id}.npz")
    if not os.path.exists(npz_path):
        missing += 1
        continue
    try:
        d = np.load(npz_path, allow_pickle=True)
        if bool(d["partial"]):
            missing += 1
    except Exception:
        missing += 1

print("OK" if missing == 0 else f"MISSING:{missing}")
PYEOF
}

shopt -s nullglob
hdf5_files=("${TELEOP_DIR}"/*.hdf5)
shopt -u nullglob

if [[ ${#hdf5_files[@]} -eq 0 ]]; then
  echo "ERROR: no .hdf5 files found in ${TELEOP_DIR}" >&2
  exit 1
fi

echo "Found ${#hdf5_files[@]} source HDF5 files in ${TELEOP_DIR}"
echo "Exporting to: ${EXPORT_DIR}"
echo ""

for hdf5_path in "${hdf5_files[@]}"; do
  base="$(basename "$hdf5_path" .hdf5)"          # e.g. heat_saucepot_safe
  task_name="${base%_safe}"
  task_name="${task_name%_unsafe}"
  sentinel="${STATUS_DIR}/.done_${base}"
  tmp_playback="${TMP_PLAYBACK_DIR}/${base}_playback.hdf5"

  if [[ -f "$sentinel" ]]; then
    echo "[${base}] SKIP (already done, sentinel present)"
    continue
  fi

  echo ""
  echo "================================================================================"
  echo "[${base}] task_name=${task_name}  source=${hdf5_path}"
  echo "================================================================================"

  ok=0
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    log_path="${REPO_ROOT}/logs/playback_${base}_attempt${attempt}.log"
    echo "[${base}] attempt ${attempt}/${MAX_ATTEMPTS} -> ${log_path}"

    timeout "$TIMEOUT_SECS" python scripts/playback_b1k.py \
      --task_name "$task_name" \
      --playback \
      --source_hdf5_path "$hdf5_path" \
      --playback_hdf5_path "$tmp_playback" \
      --export_pointworld_dir "$EXPORT_DIR" \
      --export_checkpoint_every 50 \
      > "$log_path" 2>&1
    rc=$?

    result="$(verify_export "$base" "$hdf5_path")"
    if [[ "$result" == "OK" ]]; then
      echo "[${base}] OK (rc=${rc}, attempt ${attempt})"
      touch "$sentinel"
      ok=1
      break
    else
      echo "[${base}] attempt ${attempt} incomplete (rc=${rc}, ${result}) -- tail of log:" >&2
      tail -n 20 "$log_path" >&2
    fi
  done

  if [[ "$ok" -ne 1 ]]; then
    echo "[${base}] FAILED after ${MAX_ATTEMPTS} attempts" >&2
    FAILED+=("$base")
  fi
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Finished with failures (${#FAILED[@]}):"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi

echo "All playback+export runs completed successfully."
