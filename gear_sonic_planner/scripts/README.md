# SONIC Whole-Body Trajectory Replay Commands

Replay a whole-body trajectory NPZ through the SONIC policy — the same idea as
`planner_wbc` (two terminals, press `]`), but the reference covers all 29
joints and SONIC tracks the legs too. Pass the NPZ directly; conversion
happens internally.

| | planner_wbc | SONIC replay (this) |
|---|---|---|
| Terminal 1 | `run_g1_control_loop.py` | `run_sim_loop.py` (MuJoCo sim) |
| Terminal 2 | `run_planner_policy_loop.py` | `run_wholebody_replay.py <npz>` |
| Activate | press `]` | press `]`, then `T` to play |
| Joints commanded | 17 upper (passthrough to PD) | 29 whole-body (tracked by policy) |

Run everything from the repository root. Every terminal needs `.venv_sim`
activated first (a conda `base` prompt will fail with
`ModuleNotFoundError: No module named 'gear_sonic'`).

## Simulation

Terminal 1 — MuJoCo simulator:

```bash 
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

The `[ChannelFactory] create domain error` / `Note: Channel factory
initialization attempt` lines here are harmless (a redundant second init that
is caught) — the sim publishes robot state fine. "Robot has fallen" warnings
are also normal until the controller in Terminal 2 is started with `]`.
Keyboard keys do nothing in this terminal.

Terminal 2 — replay the NPZ (converts internally, then starts SONIC):

```bash
source .venv_sim/bin/activate
python gear_sonic_planner/scripts/run_wholebody_replay.py \
    dataset/wholebody_box.npz
```

In Terminal 2, once both are up:

| Key | Action |
|-----|--------|
| `]` | Start the control system |
| `T` | Play the motion to the end |
| `I` | Reinitialize base quaternion / heading |
| `O` | Stop / exit |
| `R` or `` ` `` | Emergency stop |

If the NPZ was recorded at a rate other than 30 fps, add `--source-fps 60`.

## Real robot

```bash
source .venv_sim/bin/activate
python gear_sonic_planner/scripts/run_wholebody_replay.py \
    dataset/wholebody_box.npz --real
```

Support the robot, clear the area, and keep the physical emergency stop ready.
Test any new trajectory in simulation first — the reference is open-loop and
the legs move.

## Optional: replay the raw NPZ in mjviewer (no physics, no conversion)

```bash
source .venv_sim/bin/activate
python gear_sonic_planner/scripts/play_motion_npz.py dataset/wholebody_box.npz
```

Poses the robot with `mj_forward` only — exactly what was recorded, no policy,
no gravity. Flags: `--fps`, `--loop`, `--start/--end`, `--frame N` (hold one
pose), `--show-skeleton` (overlay recorded joint positions), `--dry-run`.

## Optional: visual check without the policy

`--convert-only` builds the reference without launching SONIC (it lands under
`gear_sonic_planner/reference/<name>/<name>/`), then view it:

```bash
source .venv_sim/bin/activate
python gear_sonic_planner/scripts/run_wholebody_replay.py \
    dataset/wholebody_box.npz --convert-only
cd gear_sonic_deploy
python visualize_motion.py \
    --motion_dir ../gear_sonic_planner/reference/wholebody_box/wholebody_box
```

Keys: `Space` pause, `R` reset, `.` step.

## Format notes (what happens internally)

- Deploy plays references at **50 Hz** (`localmotion_kplanner.hpp`); the NPZ is
  resampled from `--source-fps`.
- Joint CSV columns are **IsaacLab order**, not MuJoCo order. The comment in
  `policy_parameters.hpp` claiming MuJoCo order is wrong — verified by FK on
  the shipped example (2 mm body-position agreement under IsaacLab vs 23 cm
  under MuJoCo), and `visualize_motion.py` applies `isaaclab_to_mujoco` when
  loading.
- `body_quat.csv` is **wxyz**; body columns follow the 14-body list from
  `commands/terms/motion.yaml`; `metadata.txt` must contain the
  "Body part indexes:" line (`MotionDataReader::ReadMetadata` fails without it).
- The `--motion-data` folder handed to deploy is a *base* directory containing
  one subfolder per motion.
