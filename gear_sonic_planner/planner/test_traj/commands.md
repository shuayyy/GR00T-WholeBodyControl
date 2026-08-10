# Commands — running test trajectories

Paths below are written from the repo root:
`/home/athena/GR00T-WholeBodyControl`.

## Layout

```
test_traj/
├── grasp/                  # grasp-stage trajectories
│   ├── csv/  AORRTC  BITstar  PRM  RRTConnect  RRTstar  fixed  grasp  original
│   └── npz/  *.npz
├── original/               # planner runs on the original trajectory
│   ├── csv/  AORRTC  RRTConnect  RRTstar
│   └── npz/  *.npz
├── yaw_swap/               # CoM-valley problem (goal/yaw_*.npz) — the one
│   ├── csv/  RRTConnect    # pair that actually needs search
│   └── npz/
├── original.npz
├── traj.md                 # what each trajectory is, and how it was made
└── commands.md             # this file
```

Planner outputs carry a `degenerate_path_recovered` flag in the NPZ. The
planner now raises instead of recovering, so new files are always `False`
(genuine planner output). The existing `AORRTC.npz` files predate that change
and are recovered start→goal geodesics, not search results — see
`gear_sonic_planner/PLANNER_NOTES.md`.

Deploy plays the **`csv/`** folders. It cannot read `.npz`.

## Run one trajectory in sim

**Terminal 1** — MuJoCo simulator (start once, leave running):

```bash
cd /home/athena/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

**Terminal 2** — SONIC controller:

```bash
cd /home/athena/GR00T-WholeBodyControl/gear_sonic_deploy
pkill -9 -f g1_deploy_onnx_ref
./deploy.sh --motion-data ../gear_sonic_planner/planner/test_traj/original/csv/AORRTC/ sim
```

Replace the trailing `original/csv/AORRTC` with any other trajectory folder.

### Key sequence

| Step | Key | Where |
|---|---|---|
| 1 | `y` | deploy terminal (confirm prompt) |
| 2 | `]` | deploy terminal — start control |
| 3 | `9` | **MuJoCo window** — release elastic band, robot stands |
| 4 | `x` | deploy terminal — finger closure to 0.9 (grasp trajectories only) |
| 5 | `t` | deploy terminal — play the motion |

Also: `I` reinit heading · `O` stop · `R` or `` ` `` emergency stop.

Skipping step 3 leaves the robot hanging from the virtual crane — tracking error
looks ~4x worse than it is.

## Load a whole set and switch between them

Point `--motion-data` at the parent `csv/` dir; deploy loads every subfolder:

```bash
./deploy.sh --motion-data ../gear_sonic_planner/planner/test_traj/original/csv/ sim
```

Then `n` / `p` cycle to the next / previous motion. Press `t` after each switch.

## Kinematic preview — no policy, no physics

```bash
cd /home/athena/GR00T-WholeBodyControl/gear_sonic_deploy
source ../.venv_sim/bin/activate
python visualize_motion.py \
    --motion_dir ../gear_sonic_planner/planner/test_traj/original/csv/AORRTC/AORRTC
```

Note: `visualize_motion.py` takes the **inner** folder (`.../AORRTC/AORRTC`),
`deploy.sh` takes the **outer** one. Keys: `Space` pause · `R` reset · `.` step.

## Adding new trajectories

Drop `.npz` files into a `npz/` folder, then convert:

```bash
cd /home/athena/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic_planner/scripts/npz_to_reference.py \
    gear_sonic_planner/planner/test_traj/original/npz/ \
    --out-root gear_sonic_planner/planner/test_traj/original/csv
```

Single file also works. The converter maps joints **by name** when the NPZ has
`joint_names`, regenerates body poses and velocities by FK, and resamples to
50 Hz.

Planner outputs carry no `fps`, so they default to 50 Hz — AORRTC's 97 waypoints
play in 1.9 s, which looks fast. Stretch with `--fps`:

```bash
python gear_sonic_planner/scripts/npz_to_reference.py <npz> --fps 25   # 2x slower
```

Required NPZ keys: `joint_pos (N,29)`, `base_pos (N,3)`, `base_quat (N,4)` wxyz.
Optional: `fps`, `joint_names`.

## Real robot

Same as sim with `real` instead of `sim`, and no simulator terminal:

```bash
./deploy.sh --motion-data ../gear_sonic_planner/planner/test_traj/grasp/csv/grasp/ real
```

Support the robot, clear the area, keep the physical e-stop in hand. Test every
new trajectory in sim first — playback is open-loop and the legs move.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Address already in use` (port 5557) | `pkill -9 -f g1_deploy_onnx_ref` — Ctrl-C/Ctrl-Z leaves the binary running as a grandchild of `just` |
| `ModuleNotFoundError: gear_sonic` | wrong env — `source .venv_sim/bin/activate` |
| `[ChannelFactory] create domain error` | harmless; the sim publishes state fine |
| `Robot has fallen` before `]` | expected — no controller attached yet |
| Keys do nothing | wrong window; `]`/`t` go to the deploy terminal, `9` to the MuJoCo window |
| Motion is a single jump | sparse path that wasn't densified — regenerate; since 2026-08-10 all saved planner trajectories are densified on-manifold (BITstar/PRM included) |
