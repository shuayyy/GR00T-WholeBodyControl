# Run

Three terminals, from the repo root:

```bash
# 1. simulation + RL policy; press ] (start), wait 2 s, press 9 (release the band)
python decoupled_wbc/control/main/planner/run_g1_control_loop.py

# 2. planner server
python decoupled_wbc/control/main/planner/run_planner_server.py \
    --use-reference --reference-trajectory-path dataset/ICRA/pour/traj.npz \
    --planning-timeout 30 --ompl-planner PhaseRRTstar

# 3. request: start = demo frame 0, goal = demo last frame
python decoupled_wbc/control/main/planner/example_planner_request.py --trajectory-path dataset/ICRA/pour/traj.npz
```

Demos: `dataset/ICRA/<task>/traj.npz` (`wave`: `traj_half.npz`).

## Options

| flag | meaning |
|---|---|
| `--ompl-planner` | `RRTstar`, `RRTConnect`, `PhaseRRTstar` |
| `--planning-xml` | scene, default `simulation/envs/g1_free.xml` (others in `simulation/envs/`) |
| `--no-record` | skip saving `recordings/<time>_<planner>.npz` (robot state + sent goals + metrics) next to the reference |
| `--max-joint-step` × `--planner-frequency` | execution speed (0.025 rad × 20 Hz = 0.5 rad/s) |

## Videos

```bash
python decoupled_wbc/control/main/planner/test/render_traj.py --npz <recording.npz>                 # robot
python decoupled_wbc/control/main/planner/test/render_traj.py --npz <recording.npz> --source plan   # planned path
```
