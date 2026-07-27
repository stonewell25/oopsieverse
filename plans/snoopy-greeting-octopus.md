# pydantic ImportError on `quickstart_b1k.py`: root cause & fix

## Context

The user followed the documented BEHAVIOR-1K install flow exactly as written in
`docs/install.md`:

```bash
python install.py --new_env --behavior1k
conda activate oopsieverse_b1k
pip install -e .
python scripts/download_demos.py --sim behavior1k
python scripts/quickstart_b1k.py
```

`quickstart_b1k.py` shells out to `scripts/playback_b1k.py`, which does
`import omnigibson as og` (playback_b1k.py:48-51). That import chain starts
Isaac Sim / Omniverse Kit, which fails with:

```
ImportError: cannot import name 'collect_definitions' from 'pydantic._internal._core_utils'
```

Diagnosis (confirmed by inspecting the live `oopsieverse_b1k` conda env):

- BEHAVIOR-1K's `setup.sh` (run by `install.py`'s `install_behavior1k()`,
  install.py:57-82) pip-installs Isaac Sim 4.5.0.0 / Omniverse Kit
  106.5.0.162521 wheels directly from `pypi.nvidia.com`.
- One of those wheels, `omni.kit.pip_archive`, vendors its **own copy of
  pydantic 2.9.2** (+ `pydantic_core 2.23.4`) inside
  `.../isaacsim/extscache/omni.kit.pip_archive-*/pip_prebundle/pydantic/`.
  This bundled copy is what Kit's extension system expects to find on
  `sys.path` when it starts up.
- Separately, step 3 of the docs (`pip install -e .`, run **after**
  `install.py`) installs oopsieverse's own `pyproject.toml` deps, which
  include unpinned `open3d`. `open3d==0.19.0` requires `dash`, and the
  currently-published `dash==4.3.0` requires `mcp>=1.23.0`, which in turn
  requires `pydantic>=2.11.0` (via `pydantic-settings`). pip's resolver
  therefore **silently upgrades** the environment's top-level `pydantic` to
  `2.13.4` to satisfy `mcp` — verified by installed-file timestamps
  (`isaacsim_kernel` at 15:31:10, `pydantic-2.13.4`/`mcp-1.28.1` at
  15:39:52-55, i.e. installed later, by `pip install -e .`).
- pydantic 2.13.4 removed the internal `collect_definitions` helper from
  `pydantic._internal._core_utils`. When Kit's own bundled pydantic 2.9.2
  code (`_generate_schema.py`, `json_schema.py`,
  `_discriminated_union.py` — all still expecting `collect_definitions`)
  ends up resolving against the newer top-level pydantic 2.9.2 file that
  no longer defines it (due to both copies being on `sys.path`
  simultaneously), the import breaks.
- Root cause in one line: **`pip install -e .` transitively upgrades
  pydantic past what Isaac Sim's bundled Kit extension requires, via
  `open3d → dash → mcp → pydantic>=2.11`.** This is not a one-off local
  mistake — it will reproducibly happen to anyone following
  `docs/install.md` verbatim for `--behavior1k`, because nothing in the repo
  pins pydantic today (confirmed via repo-wide grep — no pydantic pin exists
  in `oopsieverse/pyproject.toml`, `install.py`, or anywhere under
  `externals/behavior1k`).
- `mcp`/`dash` are unrelated to BEHAVIOR-1K functionality (they come along
  only because `open3d` optionally supports a `dash`-based web visualizer
  that this project doesn't use), so forcing pydantic back down is safe.

## Fix

### 1. Unblock the current environment (run now, not part of this plan's file edits)

In the `oopsieverse_b1k` conda env, force pydantic back to the version Isaac
Sim's Kit bundle expects:

```bash
conda activate oopsieverse_b1k
pip install "pydantic==2.9.2" "pydantic-core==2.23.4"
```

This will print a pip dependency-conflict warning for `mcp`/`pydantic-settings`
(harmless — those packages aren't exercised by BEHAVIOR-1K/OmniGibson code
paths). Re-run `python scripts/quickstart_b1k.py` afterward to confirm the
`ImportError` is gone.

### 2. Fix the documented install flow so future users don't hit this

**`docs/install.md`** — step 3 ("Install OopsieVerse"): add the pydantic
re-pin immediately after `pip install -e .`, scoped to the `oopsieverse_b1k`
env only (RoboCasa's env never touches Isaac Sim/Kit, so it's unaffected):

```bash
conda activate oopsieverse_b1k      # or: oopsieverse_robocasa
pip install -e .

# BEHAVIOR-1K only: pip install -e . upgrades pydantic past what Isaac
# Sim's bundled Kit extension expects; pin it back down.
pip install "pydantic==2.9.2" "pydantic-core==2.23.4"   # oopsieverse_b1k only
```

**`docs/troubleshooting.md`** — add a new entry in the same style as the
existing "SpaceMouse" / "MediaPipe" sections, documenting the exact
`ImportError` text, the cause (one-paragraph version of the Context section
above), and the fix command from step 1.

## Verification

1. In `oopsieverse_b1k`, run `pip show pydantic pydantic-core` and confirm
   `2.9.2` / `2.23.4`.
2. Run `python scripts/quickstart_b1k.py` (or whatever minimal invocation the
   user was using) and confirm the `ImportError` no longer occurs and
   OmniGibson/Isaac Sim starts.
3. Spot check that RoboCasa's docs/install flow is untouched (no pydantic pin
   needed there, since it doesn't install Isaac Sim).
