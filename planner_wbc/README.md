Run:

```bash
python planner_wbc/control/main/teleop/run_g1_control_loop.py
python planner_wbc/control/main/teleop/run_planner_policy_loop.py --loop-trajectory
```

## DECOUPLED POLICY for the planner_wbc

- This planner_wbc is built on top of `decoupled_wbc/`.
- Right now it replays the existing trajectory dataset in the repository.
- The dataset is stored in `planner_wbc/control/dataset/`.

The planner publisher sends upper-body joint targets into the existing WBC control loop.
The current dataset contains waist and arm joints.



the dataset right noow,
`planner_wbc/control/dataset/retar_dualarm_18.npz`

-CURRENT DATASET
  `waist_yaw_joint`, `waist_roll_joint`, `waist_pitch_joint`,
  `left_shoulder_pitch_joint`, `left_shoulder_roll_joint`, `left_shoulder_yaw_joint`,
  `left_elbow_joint`, `left_wrist_roll_joint`, `left_wrist_pitch_joint`, `left_wrist_yaw_joint`,
  `right_shoulder_pitch_joint`, `right_shoulder_roll_joint`, `right_shoulder_yaw_joint`,
  `right_elbow_joint`, `right_wrist_roll_joint`, `right_wrist_pitch_joint`, `right_wrist_yaw_joint`
