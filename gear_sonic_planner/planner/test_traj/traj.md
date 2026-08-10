# Test trajectories — `wholebody_box`

Three whole-body G1 trajectories, each derived from the previous one. All are
**299 frames @ 50 Hz** (5.98 s), a box pick-up: stand → bend/reach → lift →
stand holding.

Source data: `dataset/wholebody_box.npz` (motionbricks 34-joint `G1Skeleton34`
capture, 180 frames @ 30 fps, y-up/z-forward).

| File | What it is |
|---|---|
| `original.npz` | Straight conversion of the capture. Self-colliding. |
| `fixed.npz` | Collision-free. Shoulder-roll only. |
| `grasp.npz` | Symmetric two-hand box grasp with enforced clearances. |

## NPZ contents

Common keys: `joint_pos (299, 29)`, `base_pos (299, 3)`,
`base_quat (299, 4)` wxyz, `joint_names (29,)`, `fps = 50.0`,
`joint_order = "mujoco"`.

`grasp.npz` additionally has `finger_closure`, `finger_targets_left`,
`ee_body_clearance_m`, `ee_ee_clearance_m`.

> **Joint order is MuJoCo** in these NPZ files. The deploy CSV references use
> **IsaacLab** order — convert with `MJ_TO_IL` from
> `gear_sonic_planner/scripts/run_wholebody_replay.py`. Getting this wrong loads
> fine but scrambles the limbs.

## What each fix did

### `original.npz`
Straight conversion, verified against the capture: MuJoCo FK reproduces the
recorded joint positions to **0.6 mm median** (6.4 mm max). Geometrically
faithful, but self-colliding:

- thumbs resting inside the thighs while the arms hang: up to **54 mm**
- left fingers ↔ right fingers interpenetrating during the carry (holding a
  phantom box): up to **27 mm**

The 29-joint body trajectory itself is nearly clean — worst body-body contact is
a 4 mm wrist–hip graze. Nearly all the violation comes from the **fingers at
their frozen open pose** (the reference has no hand joints).

### `fixed.npz` — collision-free
Iterative projection: push each arm outboard via **shoulder roll only**, the
minimum needed to clear every contact, then dilate + smooth the correction so
velocities stay continuous. Converged in 6 iterations.

| | |
|---|---|
| Self-collisions | **0 frames, 0.00 mm** |
| Joints changed | **2 / 29** (left + right shoulder roll) |
| Max change | 0.204 rad (11.7°) |
| Mean abs change over all data | 0.0046 rad |
| EE shift vs original | left 5.2 cm max, right 7.3 cm max — almost entirely lateral (y) |

Clearances afterwards: **EE↔EE 21.7 mm**, tightest overall pair (thumb vs hip)
**0.1 mm** — collision-free but grazing, essentially no safety margin.

### `grasp.npz` — symmetric grasp
Mirror-average the two wrist poses in the pelvis frame, offset laterally +
forward, then damped-least-squares IK on each 7-DOF arm chain. Clearances
measured with fingers at 90 % closure.

| Requirement | Achieved |
|---|---|
| EE → body ≥ 10 cm | **100.1 mm** |
| EE → EE ≥ 10 cm | **106.2 mm** |
| Palms opposite-parallel | mirror rotation residual **0.09°** |
| Fingers 90 % closed | 0.9 × the deploy's full-close pose |
| Both EE same z | **not exact** — mean 12 mm, max 42 mm on 70/299 frames |

Settings: `y_half = 0.18 m` (wrists 36 cm apart), forward push 0.170–0.174 m.
14 / 29 joints changed, max delta 2.4 rad — a much larger rework than `fixed`.

**Why z isn't exact:** once both 10 cm clearances hold, the perfectly mirrored
target is outside the arm's reachable set on the deep-squat frames (140 mm IK
position residual there). Not a solver issue — no joints at limits, and damping
was swept 1e-2 → 1e-5 with up to 600 iterations. To get z exact, one of the
other constraints has to give.

**Geometry note:** a 106 mm fingertip gap requires wrists 36 cm apart, because
90 %-closed fingers curl inward and eat most of the span. Forced by the hand
geometry, not a choice.

## Related files

| Path | What |
|---|---|
| `gear_sonic_planner/scripts/run_wholebody_replay.py` | NPZ → deploy reference, then launches SONIC |
| `gear_sonic_planner/scripts/play_motion_npz.py` | Kinematic mjviewer replay (no physics) |
| `gear_sonic_planner/scripts/make_grasp_reference.py` | Built `grasp` from `fixed` |
| `gear_sonic_planner/reference/wholebody_box{,_fixed,_grasp}/` | Deploy-ready CSV references |
| `gear_sonic_planner/planner/goal/start.npz`, `goal.npz` | Frame 0 and frame 122 of `grasp`, 17 upper-body joints |
| `test_dataset/deployed_trajectory.npz` | What the robot **actually did** tracking `fixed` in sim |
| `test_dataset/comparison_results.md` | Reference-vs-deployed tracking report |
| `debugclause/` | Videos + EE/pose graphs of `fixed` |

## Tracking result (for context)

Sim run of `fixed` through the SONIC policy, elastic band released:
**4.9° mean joint error**, waveform correlation 0.83, squats to 0.546 m vs the
reference 0.460 m. Error doubles inside the frames where the reference was
originally self-colliding.

Note: with the elastic band still attached the same comparison gave 20.8° — the
band, not the policy, was the dominant error source. Release it with `9` in the
sim window before judging tracking.
