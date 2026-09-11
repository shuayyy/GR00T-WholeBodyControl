# Simulation versus hardware, PhaseRRT*

Six retargeted demonstrations, planned and executed once each in MuJoCo and once each on the Unitree G1.
Both sides ran the identical command apart from `--interface real` and `--no-visualize-planning`:
PhaseRRT* with a 30 s budget, `simulation/envs/g1_free.xml` with collision checking on, 0.025 rad per
waypoint at 20 Hz (0.5 rad/s), 5 s smoothstep ramps, recording on, no path smoothing (`smooth_path`
was off; it is now the default, so pass `--no-smooth-path` to reproduce).

Data: `sim_test_final/<demo>/` and `real/<demo>/`, each holding the run and the demo it used.

## Metrics

- **Tracking error**: how far the robot's joints lag behind the commanded joints (rad), over the 17 upper-body joints and the streamed window only.
- **Planned deviation**: how far the planned path is from the demonstration, arc-length matched (rad, and mm at the right wrist).
- **Deployed deviation**: how far the executed motion is from the demonstration, same matching.
- **EE tracking**: right-wrist distance between the robot and the commanded pose (mm).

## Per demo

| demo | waypoints | window (s) | tracking sim / real | planned dev sim / real | deployed dev sim / real |
|---|---|---|---|---|---|
| dual arm sweep | 112 / 114 | 5.5 / 5.6 | 0.041 / 0.041 | 0.008 / 0.006 | 0.038 / 0.037 |
| handover | 90 / 87 | 4.4 / 4.2 | 0.033 / 0.035 | 0.005 / 0.006 | 0.033 / 0.034 |
| pass | 61 / 62 | 2.9 / 3.0 | 0.037 / 0.035 | 0.011 / 0.008 | 0.042 / 0.038 |
| pour | 98 / 98 | 4.8 / 4.8 | 0.033 / 0.033 | 0.021 / 0.019 | 0.043 / 0.045 |
| single arm sweep | 109 / 110 | 5.3 / 5.4 | 0.037 / 0.036 | 0.005 / 0.006 | 0.034 / 0.033 |
| wave | 291 / 292 | 14.4 / 14.5 | 0.038 / 0.036 | 0.014 / 0.018 | 0.034 / 0.032 |
| **average** | | | **0.036 / 0.036** | **0.011 / 0.010** | **0.037 / 0.037** |

## Right wrist, millimetres

| demo | planned dev sim / real | deployed dev sim / real | EE tracking sim / real |
|---|---|---|---|
| dual arm sweep | 6 / 5 | 51 / 50 | 53 / 53 |
| handover | 4 / 3 | 60 / 73 | 64 / 78 |
| pass | 11 / 7 | 66 / 74 | 61 / 69 |
| pour | 22 / 19 | 73 / 71 | 66 / 60 |
| single arm sweep | 6 / 8 | 57 / 58 | 60 / 64 |
| wave | 17 / 23 | 100 / 77 | 103 / 87 |
| **average** | **11 / 11** | **68 / 67** | **68 / 68** |

## Summary

| quantity | simulation | hardware | difference |
|---|---|---|---|
| Tracking error (rad) | 0.036 | 0.036 | ~0 |
| Planned joint deviation (rad) | 0.011 | 0.010 | ~0 |
| Deployed joint deviation (rad) | 0.037 | 0.037 | 0.001 |
| Planned EE deviation (mm) | 11 | 11 | ~0 |
| Deployed EE deviation (mm) | 68 | 67 | 1 |
| EE tracking (mm) | 68 | 68 | 1 |
| Command-to-robot lag (s) | 0.16 | 0.18 | ~0 |
| Tracking with lag removed (rad) | 0.032 | 0.031 | 0.001 |
