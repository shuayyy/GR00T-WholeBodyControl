# gear_sonic/planner

Constrained whole-body planner for SONIC: plans 29-joint trajectories
(feet pinned, CoM-stable) and streams them to the C++ deploy .

## Setup

Every terminal:

```bash
source .venv_sim/bin/activate
```


## Run 

```bash
# 1 — MuJoCo sim
python gear_sonic/scripts/run_sim_loop.py

# 2 — SONIC policy
cd gear_sonic_deploy && ./deploy.sh --input-type zmq_manager sim
#   press 9 in the MuJoCo window to release the band

# 3 — planner server (switches the deploy to streamed-motion mode on startup)
python gear_sonic/planner/run_planner_server.py
#   NOW press ] in the deploy terminal to start control (O stops)

# 4 — one plan request (standing -> box-lift pose by default)
python gear_sonic/planner/example_planner_request.py
```

## Speed control

```bash
python gear_sonic/planner/run_planner_server.py \
    --velocity.max-joint-velocity 0.2 \   # rad/s cap (default 0.5)
    --velocity.max-base-velocity 0.1 \    # m/s cap (default 0.25)
    --velocity.time-scale 0.5             # 0.5 = half speed (default 1.0)
```

