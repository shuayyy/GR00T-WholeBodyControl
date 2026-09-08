# Deployment checklist

One rehearsal in simulation, then the same run on the robot. Every terminal starts with:

```bash
cd ~/GR00T-WholeBodyControl && source .venv_sim/bin/activate
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH="$PWD/.venv_sim/lib/python3.10/site-packages/ompl:$LD_LIBRARY_PATH"
```

## What a run looks like

The server executes a plan in four stages and asks for Enter before each motion:

```
ramp up  ->  execute  ->  save recording + metrics  ->  ramp down
 (Enter)     (Enter)                                    (Enter)
```

- **ramp up**: from wherever the arms are to the plan's first waypoint, smoothstep.
- **execute**: the plan at 0.5 rad/s (0.025 rad x 20 Hz).
- **save**: `Saved recording ...` and the metrics print. The return motion is not part of the data.
- **ramp down**: back to the pose the arms were in before the ramp up.

Each prompt prints the largest joint gap it is about to cover. Ctrl-C at a prompt stops there; the arms hold.

## 1. Simulation rehearsal

Terminal 1, then press `]` (the arms glide to the default pose over 5 s), wait, press `9` in the MuJoCo window:

```bash
python decoupled_wbc/control/main/planner/run_g1_control_loop.py
```

Terminal 2, wait for `Planner service ready`:

```bash
python decoupled_wbc/control/main/planner/run_planner_server.py \
    --use-reference --reference-trajectory-path dataset/ICRA/pour/traj.npz \
    --planning-timeout 30 --ompl-planner PhaseRRTstar
```

Terminal 3:

```bash
python decoupled_wbc/control/main/planner/example_planner_request.py --trajectory-path dataset/ICRA/pour/traj.npz
```

Press Enter at the three prompts in terminal 2. Expect: tracking mean about 0.04 rad, plan right-wrist deviation about 7 mm, `Run complete`. Then:

```bash
NPZ=$(ls -t decoupled_wbc/control/main/planner/dataset/ICRA/pour/recordings/*_PhaseRRTstar.npz | head -1)
python decoupled_wbc/control/main/planner/test/render_traj.py --npz $NPZ
python decoupled_wbc/control/main/planner/test/render_traj.py --npz $NPZ --source plan
```

## 2. Hardware

Robot on the hoist, E-stop in hand, nothing within reach of the arms (the waist turns too).
PC on the Unitree network: `ping 192.168.123.164`.

Terminal 1. The arms stay where they are at launch (the policy is seeded from the measured pose).
Press `]` only; there is no `9` on hardware. `]` starts the balance policy and glides the arms
to the default pose (arms down) over 5 s; wait for that to finish, then lower the robot to the floor.

```bash
python decoupled_wbc/control/main/planner/run_g1_control_loop.py --interface real
```

Check state is flowing (about 50 Hz):

```bash
ros2 topic hz /G1Env/env_state_act
```

Terminal 2. Headless PC, 8 s ramps, and half speed for the first run:

```bash
python decoupled_wbc/control/main/planner/run_planner_server.py \
    --no-visualize-planning --initial-transition-time 8 --max-joint-step 0.01 \
    --use-reference --reference-trajectory-path dataset/ICRA/pour/traj.npz \
    --planning-timeout 30 --ompl-planner PhaseRRTstar
```

Terminal 3, plan without moving first and read the endpoint errors:

```bash
python decoupled_wbc/control/main/planner/example_planner_request.py \
    --trajectory-path dataset/ICRA/pour/traj.npz --no-execute-immediately
```

Then the real request, and answer the three prompts in terminal 2 while watching the robot:

```bash
python decoupled_wbc/control/main/planner/example_planner_request.py --trajectory-path dataset/ICRA/pour/traj.npz
```

Once a run is clean, drop `--max-joint-step 0.01` to get the 0.5 rad/s used for all simulation results.

## Rules while it runs

- Wait for `Saved recording ...` before Ctrl-C. Interrupting earlier loses that run's data.
- Do not send a second request while a motion is streaming; planning blocks the goal stream for 30 s and the controller's 1 s watchdog fires.
- Ctrl-C is not an emergency stop. After the run it does nothing (arms hold). Mid-motion the arms keep moving for about 1 s (0.5 rad at 0.5 rad/s) before they stop. Use the E-stop.
- If terminal 1 exits for any reason, the robot is uncommanded. Keep it on the hoist or have a catcher.

## Stop safely (the order used on the robot before)

1. Ctrl-C terminal 2 (server). The arms hold.
2. Support the robot on the hoist.
3. In terminal 1 press `o`: the balance policy hands the legs back to a position hold at their current angles.
4. Ctrl-C terminal 1.

## Known differences on hardware

- `base_pose` in the recording is `[0, 0, 0]` plus the IMU quaternion; the robot reports no position. `render_traj.py` will draw the robot at floor level. Tracking and deviation metrics do not use it and are unaffected.
- The tracking error compares the robot with the goal current at that instant, not lag-compensated, same convention as the simulation numbers. `goal_target_time` is in the npz if you want to recompute.
- The three waist joints in the tracking error measure the balance policy, which owns the waist, not the arm controller.
- Every goal sets the 14 finger joints to the controller default (open). Do not hold an object.
- The planning scene is `g1_free.xml`, no obstacles. With a real table use `--planning-xml simulation/envs/g1_table.xml` and check its position; it sits 10 cm further back than the lab scan.
